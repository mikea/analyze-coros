use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use gimli::{DwarfSections, EndianSlice, LittleEndian, Reader, UnitOffset};
use memmap2::Mmap;
use object::{Object, ObjectSection, ObjectSymbol};
use owo_colors::OwoColorize;
use std::{borrow::Cow, collections::HashMap, fs::File, path::PathBuf};
use tabled::{settings::Style, Table, Tabled};

#[derive(Parser)]
#[command(name = "analyze-coros")]
#[command(about = "Analyze C++ coroutines in compiled binaries")]
struct Args {
    /// Path to the compiled binary to analyze
    bin: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Show detailed information about a specific coroutine
    Details {
        /// Coroutine name or location (e.g., "coroFib10" or "async-bench.c++:253")
        name: String,
    },
}

/// Information about a coroutine function (ramp, resume, or destroy)
#[derive(Debug, Clone)]
struct FunctionInfo {
    /// Address of the function in the binary
    address: u64,
    /// Size of the function in bytes
    size: u64,
    /// Demangled name (if available from DWARF)
    demangled_name: Option<String>,
    /// Source file (if available from DWARF)
    file: Option<String>,
    /// Source line (if available from DWARF)
    line: Option<u64>,
}

/// A C++ coroutine with its three component functions
#[derive(Debug)]
struct Coroutine {
    /// Mangled linkage name (e.g., _Z9coroFib10m)
    linkage_name: String,
    /// The ramp function (initial entry point that creates the coroutine frame)
    ramp: Option<FunctionInfo>,
    /// The resume function (called to resume the coroutine)
    resume: Option<FunctionInfo>,
    /// The destroy function (called to destroy the coroutine frame)
    destroy: Option<FunctionInfo>,
}

impl Coroutine {
    fn new(linkage_name: String) -> Self {
        Coroutine {
            linkage_name,
            ramp: None,
            resume: None,
            destroy: None,
        }
    }

    fn display_name(&self) -> String {
        self.ramp
            .as_ref()
            .and_then(|f| f.demangled_name.as_ref())
            .or_else(|| self.resume.as_ref().and_then(|f| f.demangled_name.as_ref()))
            .or_else(|| {
                self.destroy
                    .as_ref()
                    .and_then(|f| f.demangled_name.as_ref())
            })
            .cloned()
            .unwrap_or_else(|| self.linkage_name.clone())
    }

    fn source_location(&self) -> String {
        self.ramp
            .as_ref()
            .and_then(|f| {
                let filename = f.file.as_ref().map(|p| p.rsplit('/').next().unwrap_or(p));
                match (filename, f.line) {
                    (Some(file), Some(line)) => Some(format!("{}:{}", file, line)),
                    (Some(file), None) => Some(file.to_string()),
                    _ => None,
                }
            })
            .unwrap_or_default()
    }

    fn matches(&self, query: &str) -> bool {
        self.display_name() == query || self.source_location() == query
    }
}

/// Row for the coroutines table
#[derive(Tabled)]
struct CoroutineRow {
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Location")]
    location: String,
}

/// DWARF info extracted from a subprogram entry
#[derive(Clone)]
struct DwarfFunctionInfo {
    name: Option<String>,
    file: Option<String>,
    line: Option<u64>,
}

fn load_section<'a>(
    object: &'a object::File<'a>,
    section_name: &str,
) -> Result<Cow<'a, [u8]>, gimli::Error> {
    Ok(object
        .section_by_name(section_name)
        .map(|s| s.uncompressed_data().unwrap_or(Cow::Borrowed(&[])))
        .unwrap_or(Cow::Borrowed(&[])))
}

/// Find all coroutines by looking for symbols with .resume and .destroy suffixes
fn find_coroutines(object: &object::File) -> HashMap<String, Coroutine> {
    let mut coroutines: HashMap<String, Coroutine> = HashMap::new();

    for symbol in object.symbols() {
        let name = match symbol.name() {
            Ok(n) => n,
            Err(_) => continue,
        };

        let address = symbol.address();
        let size = symbol.size();

        // Check for .resume suffix
        if let Some(base_name) = name.strip_suffix(".resume") {
            let coro = coroutines
                .entry(base_name.to_string())
                .or_insert_with(|| Coroutine::new(base_name.to_string()));
            coro.resume = Some(FunctionInfo {
                address,
                size,
                demangled_name: None,
                file: None,
                line: None,
            });
        }
        // Check for .destroy suffix
        else if let Some(base_name) = name.strip_suffix(".destroy") {
            let coro = coroutines
                .entry(base_name.to_string())
                .or_insert_with(|| Coroutine::new(base_name.to_string()));
            coro.destroy = Some(FunctionInfo {
                address,
                size,
                demangled_name: None,
                file: None,
                line: None,
            });
        }
    }

    // Now find the ramp functions (base name without suffix)
    for symbol in object.symbols() {
        let name = match symbol.name() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Skip if it has a suffix
        if name.contains(".resume") || name.contains(".destroy") {
            continue;
        }

        // Check if this is a ramp function for a known coroutine
        if let Some(coro) = coroutines.get_mut(name) {
            coro.ramp = Some(FunctionInfo {
                address: symbol.address(),
                size: symbol.size(),
                demangled_name: None,
                file: None,
                line: None,
            });
        }
    }

    coroutines
}

/// Extract function info from a DIE, following DW_AT_specification if needed
fn extract_function_info<R: Reader>(
    dwarf: &gimli::Dwarf<R>,
    unit: &gimli::Unit<R>,
    entry: &gimli::DebuggingInformationEntry<R>,
    die_cache: &HashMap<UnitOffset<R::Offset>, DwarfFunctionInfo>,
) -> DwarfFunctionInfo {
    // First try to get info directly from this entry
    let mut name = entry.attr(gimli::DW_AT_name).and_then(|attr| {
        dwarf
            .attr_string(unit, attr.value())
            .ok()
            .and_then(|s| s.to_string_lossy().ok().map(|cow| cow.into_owned()))
    });

    let mut file = extract_file(dwarf, unit, entry);
    let mut line = entry
        .attr(gimli::DW_AT_decl_line)
        .and_then(|attr| attr.udata_value());

    // If we have a specification reference, get info from there
    if let Some(spec_attr) = entry.attr(gimli::DW_AT_specification) {
        if let gimli::AttributeValue::UnitRef(offset) = spec_attr.value() {
            if let Some(spec_info) = die_cache.get(&offset) {
                if name.is_none() {
                    name = spec_info.name.clone();
                }
                if file.is_none() {
                    file = spec_info.file.clone();
                }
                if line.is_none() {
                    line = spec_info.line;
                }
            }
        }
    }

    // Also check DW_AT_abstract_origin (used for inlined functions)
    if let Some(origin_attr) = entry.attr(gimli::DW_AT_abstract_origin) {
        if let gimli::AttributeValue::UnitRef(offset) = origin_attr.value() {
            if let Some(origin_info) = die_cache.get(&offset) {
                if name.is_none() {
                    name = origin_info.name.clone();
                }
                if file.is_none() {
                    file = origin_info.file.clone();
                }
                if line.is_none() {
                    line = origin_info.line;
                }
            }
        }
    }

    DwarfFunctionInfo { name, file, line }
}

fn extract_file<R: Reader>(
    dwarf: &gimli::Dwarf<R>,
    unit: &gimli::Unit<R>,
    entry: &gimli::DebuggingInformationEntry<R>,
) -> Option<String> {
    entry.attr(gimli::DW_AT_decl_file).and_then(|attr| {
        if let gimli::AttributeValue::FileIndex(idx) = attr.value() {
            if let Some(ref line_program) = unit.line_program {
                let header = line_program.header();
                if let Some(file_entry) = header.file(idx) {
                    if let Some(dir) = file_entry.directory(header) {
                        let dir_str = dwarf.attr_string(unit, dir).ok()?;
                        let file_str = dwarf.attr_string(unit, file_entry.path_name()).ok()?;
                        return Some(format!(
                            "{}/{}",
                            dir_str.to_string_lossy().ok()?,
                            file_str.to_string_lossy().ok()?
                        ));
                    } else {
                        let file_str = dwarf.attr_string(unit, file_entry.path_name()).ok()?;
                        return file_str.to_string_lossy().ok().map(|s| s.into_owned());
                    }
                }
            }
        }
        None
    })
}

/// Enrich coroutine data with DWARF debug information
fn enrich_with_dwarf<R: Reader>(
    coroutines: &mut HashMap<String, Coroutine>,
    dwarf: &gimli::Dwarf<R>,
) -> Result<()> {
    // Build a map of address -> function info from DWARF
    let mut addr_info: HashMap<u64, DwarfFunctionInfo> = HashMap::new();

    let mut iter = dwarf.units();
    while let Some(header) = iter.next()? {
        let unit = dwarf.unit(header)?;

        // First pass: build a cache of all DIEs that might be referenced
        let mut die_cache: HashMap<UnitOffset<R::Offset>, DwarfFunctionInfo> = HashMap::new();
        let mut entries = unit.entries();
        while let Some(entry) = entries.next_dfs()? {
            if entry.tag() == gimli::DW_TAG_subprogram {
                let offset = entry.offset();
                let name = if let Some(attr) = entry.attr(gimli::DW_AT_name) {
                    dwarf
                        .attr_string(&unit, attr.value())
                        .ok()
                        .and_then(|s| s.to_string_lossy().ok().map(|cow| cow.into_owned()))
                } else {
                    None
                };
                let file = extract_file(dwarf, &unit, entry);
                let line = entry
                    .attr(gimli::DW_AT_decl_line)
                    .and_then(|attr| attr.udata_value());

                die_cache.insert(offset, DwarfFunctionInfo { name, file, line });
            }
        }

        // Second pass: extract function info, resolving references
        let mut entries = unit.entries();
        while let Some(entry) = entries.next_dfs()? {
            if entry.tag() == gimli::DW_TAG_subprogram {
                // Get the low_pc (function address)
                let low_pc = if let Some(attr) = entry.attr(gimli::DW_AT_low_pc) {
                    match attr.value() {
                        gimli::AttributeValue::Addr(addr) => Some(addr),
                        gimli::AttributeValue::DebugAddrIndex(idx) => {
                            dwarf.address(&unit, idx).ok()
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                if let Some(addr) = low_pc {
                    let info = extract_function_info(dwarf, &unit, entry, &die_cache);
                    addr_info.insert(addr, info);
                }
            }
        }
    }

    // Now enrich the coroutines
    for coro in coroutines.values_mut() {
        if let Some(ref mut ramp) = coro.ramp {
            if let Some(info) = addr_info.get(&ramp.address) {
                ramp.demangled_name = info.name.clone();
                ramp.file = info.file.clone();
                ramp.line = info.line;
            }
        }
        if let Some(ref mut resume) = coro.resume {
            if let Some(info) = addr_info.get(&resume.address) {
                resume.demangled_name = info.name.clone();
                resume.file = info.file.clone();
                resume.line = info.line;
            }
        }
        if let Some(ref mut destroy) = coro.destroy {
            if let Some(info) = addr_info.get(&destroy.address) {
                destroy.demangled_name = info.name.clone();
                destroy.file = info.file.clone();
                destroy.line = info.line;
            }
        }
    }

    Ok(())
}

/// Load binary and extract coroutines with DWARF info
fn load_coroutines(bin: &PathBuf) -> Result<Vec<Coroutine>> {
    // Open and memory-map the binary
    let file =
        File::open(bin).with_context(|| format!("Failed to open binary: {}", bin.display()))?;

    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("Failed to memory-map binary: {}", bin.display()))?;

    // Parse the binary
    let object = object::File::parse(&*mmap)
        .with_context(|| format!("Failed to parse binary: {}", bin.display()))?;

    // Find coroutines from symbol table
    let mut coroutines = find_coroutines(&object);

    if coroutines.is_empty() {
        return Ok(Vec::new());
    }

    // Load DWARF sections
    let load_section_fn = |id: gimli::SectionId| -> Result<Cow<[u8]>, gimli::Error> {
        load_section(&object, id.name())
    };

    let debug_info_data = load_section_fn(gimli::SectionId::DebugInfo)?;
    if debug_info_data.is_empty() {
        eprintln!(
            "{} No DWARF debug information found in binary: {}",
            "Warning:".yellow(),
            bin.display()
        );
        eprintln!(
            "{} Compile with -g flag to include debug information",
            "Hint:".cyan()
        );
    } else {
        // Create the DWARF context and enrich coroutines
        let dwarf_sections = DwarfSections::load(&load_section_fn)?;
        let dwarf = dwarf_sections.borrow(|section| EndianSlice::new(section, LittleEndian));
        enrich_with_dwarf(&mut coroutines, &dwarf)?;
    }

    let mut result: Vec<_> = coroutines.into_values().collect();
    result.sort_by(|a, b| a.display_name().cmp(&b.display_name()));
    Ok(result)
}

/// Print the list of coroutines
fn cmd_list(coroutines: &[Coroutine]) {
    if coroutines.is_empty() {
        println!("No coroutines found in binary.");
        return;
    }

    println!("{}", "# Coroutines".bold());
    println!();

    let rows: Vec<CoroutineRow> = coroutines
        .iter()
        .map(|coro| CoroutineRow {
            name: coro.display_name(),
            location: coro.source_location(),
        })
        .collect();

    let table = Table::new(rows).with(Style::sharp()).to_string();
    println!("{}", table);

    println!();
    println!(
        "{} Use `{} <name|location>` to see coroutine details",
        "Hint:".cyan(),
        "details".green()
    );
}

/// Print details for a specific coroutine
fn cmd_details(coroutines: &[Coroutine], query: &str) -> Result<()> {
    let matches: Vec<_> = coroutines.iter().filter(|c| c.matches(query)).collect();

    if matches.is_empty() {
        eprintln!("{} No coroutine found matching '{}'", "Error:".red(), query);
        eprintln!();
        eprintln!("Available coroutines:");
        for coro in coroutines {
            eprintln!("  - {} ({})", coro.display_name(), coro.source_location());
        }
        bail!("Coroutine not found");
    }

    if matches.len() > 1 {
        eprintln!(
            "{} Multiple coroutines match '{}'. Use location to disambiguate:",
            "Error:".red(),
            query
        );
        eprintln!();
        for coro in &matches {
            eprintln!(
                "  {} details {}",
                "->".cyan(),
                coro.source_location().green()
            );
        }
        bail!("Ambiguous coroutine name");
    }

    let coro = matches[0];

    println!("{}", "# Coroutine Details".bold());
    println!();

    println!("{:<12} {}", "Name:".bold(), coro.display_name().green());
    println!(
        "{:<12} {}",
        "Location:".bold(),
        coro.source_location().cyan()
    );
    println!("{:<12} {}", "Linkage:".bold(), coro.linkage_name.dimmed());

    println!();
    println!("{}", "## Functions".bold());
    println!();

    if let Some(ref ramp) = coro.ramp {
        println!("{}", "Ramp (entry point):".yellow());
        println!("  {:<10} 0x{:x}", "Address:".dimmed(), ramp.address);
        println!("  {:<10} {} bytes", "Size:".dimmed(), ramp.size);
    }

    if let Some(ref resume) = coro.resume {
        println!("{}", "Resume:".yellow());
        println!("  {:<10} 0x{:x}", "Address:".dimmed(), resume.address);
        println!("  {:<10} {} bytes", "Size:".dimmed(), resume.size);
    }

    if let Some(ref destroy) = coro.destroy {
        println!("{}", "Destroy:".yellow());
        println!("  {:<10} 0x{:x}", "Address:".dimmed(), destroy.address);
        println!("  {:<10} {} bytes", "Size:".dimmed(), destroy.size);
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let coroutines = load_coroutines(&args.bin)?;

    match args.command {
        None => cmd_list(&coroutines),
        Some(Command::Details { name }) => cmd_details(&coroutines, &name)?,
    }

    Ok(())
}
