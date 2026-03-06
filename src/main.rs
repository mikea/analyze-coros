use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use gimli::{DwarfSections, EndianSlice, LittleEndian, Reader, ReaderOffset, UnitOffset};
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
    address: u64,
    size: u64,
    demangled_name: Option<String>,
    file: Option<String>,
    line: Option<u64>,
}

/// A member of the coroutine frame structure
#[derive(Debug, Clone)]
struct FrameMember {
    name: String,
    type_name: String,
    offset: u64,
    size: Option<u64>,
    size_estimated: bool,
    is_artificial: bool,
    nested_members: Option<Vec<FrameMember>>,
}

/// The coroutine frame structure
#[derive(Debug, Clone)]
struct FrameInfo {
    type_name: String,
    size: u64,
    members: Vec<FrameMember>,
}

/// A C++ coroutine with its three component functions
#[derive(Debug)]
struct Coroutine {
    linkage_name: String,
    ramp: Option<FunctionInfo>,
    resume: Option<FunctionInfo>,
    destroy: Option<FunctionInfo>,
    frame: Option<FrameInfo>,
}

impl Coroutine {
    fn new(linkage_name: String) -> Self {
        Self {
            linkage_name,
            ramp: None,
            resume: None,
            destroy: None,
            frame: None,
        }
    }

    fn display_name(&self) -> &str {
        self.ramp
            .as_ref()
            .and_then(|f| f.demangled_name.as_deref())
            .unwrap_or(&self.linkage_name)
    }

    fn location(&self) -> Option<String> {
        self.ramp.as_ref().and_then(|f| {
            f.file.as_ref().map(|file| {
                let filename = std::path::Path::new(file)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| file.clone());
                match f.line {
                    Some(line) => format!("{}:{}", filename, line),
                    None => filename,
                }
            })
        })
    }

    fn frame_size(&self) -> Option<u64> {
        self.frame.as_ref().map(|f| f.size)
    }

    fn frame_type_name(&self) -> String {
        format!("{}.coro_frame_ty", self.linkage_name)
    }
}

/// DWARF function info extracted from debug info
struct DwarfFunctionInfo {
    name: Option<String>,
    file: Option<String>,
    line: Option<u64>,
}

/// DWARF analyzer with cached type definitions
struct DwarfAnalyzer<'a, R: Reader> {
    dwarf: &'a gimli::Dwarf<R>,
    units: Vec<gimli::Unit<R>>,
    /// Map from type name to (unit_index, offset, size)
    type_defs: HashMap<String, (usize, u64, Option<u64>)>,
}

impl<'a, R: Reader> DwarfAnalyzer<'a, R> {
    fn new(dwarf: &'a gimli::Dwarf<R>) -> Result<Self> {
        let mut units = Vec::new();
        let mut type_defs = HashMap::new();

        // Load all units and build type definition cache
        let mut iter = dwarf.units();
        let mut unit_index = 0;
        while let Some(header) = iter.next()? {
            let unit = dwarf.unit(header)?;

            // Scan for type definitions
            let mut entries = unit.entries();
            while let Some(entry) = entries.next_dfs()? {
                if entry.tag() == gimli::DW_TAG_structure_type
                    || entry.tag() == gimli::DW_TAG_class_type
                    || entry.tag() == gimli::DW_TAG_union_type
                {
                    // Skip declarations
                    let is_declaration = entry
                        .attr(gimli::DW_AT_declaration)
                        .map(|a| matches!(a.value(), gimli::AttributeValue::Flag(true)))
                        .unwrap_or(false);

                    if is_declaration {
                        continue;
                    }

                    if let Some(name_attr) = entry.attr(gimli::DW_AT_name) {
                        if let Ok(name) = dwarf.attr_string(&unit, name_attr.value()) {
                            if let Ok(name_str) = name.to_string_lossy() {
                                let size = entry
                                    .attr(gimli::DW_AT_byte_size)
                                    .and_then(|a| a.udata_value());
                                let offset: u64 = entry.offset().0.into_u64();
                                type_defs.insert(name_str.into_owned(), (unit_index, offset, size));
                            }
                        }
                    }
                }
            }

            units.push(unit);
            unit_index += 1;
        }

        Ok(Self {
            dwarf,
            units,
            type_defs,
        })
    }

    fn get_type_size(
        &self,
        unit: &gimli::Unit<R>,
        type_offset: UnitOffset<R::Offset>,
    ) -> Option<u64> {
        let mut cursor = unit.entries_at_offset(type_offset).ok()?;
        cursor.next_entry().ok()?;
        let entry = cursor.current()?;

        // Try to get size directly
        if let Some(size) = entry
            .attr(gimli::DW_AT_byte_size)
            .and_then(|a| a.udata_value())
        {
            return Some(size);
        }

        // For pointer/reference types, assume 8 bytes (64-bit)
        if entry.tag() == gimli::DW_TAG_pointer_type
            || entry.tag() == gimli::DW_TAG_reference_type
            || entry.tag() == gimli::DW_TAG_rvalue_reference_type
        {
            return Some(8);
        }

        // Follow const/volatile/typedef to underlying type
        if entry.tag() == gimli::DW_TAG_const_type
            || entry.tag() == gimli::DW_TAG_volatile_type
            || entry.tag() == gimli::DW_TAG_typedef
            || entry.tag() == gimli::DW_TAG_restrict_type
            || entry.tag() == gimli::DW_TAG_atomic_type
        {
            if let Some(type_attr) = entry.attr(gimli::DW_AT_type) {
                if let gimli::AttributeValue::UnitRef(inner_offset) = type_attr.value() {
                    return self.get_type_size(unit, inner_offset);
                }
            }
            return None;
        }

        // For array types with byte_size
        if entry.tag() == gimli::DW_TAG_array_type {
            if let Some(size) = entry
                .attr(gimli::DW_AT_byte_size)
                .and_then(|a| a.udata_value())
            {
                return Some(size);
            }
        }

        // Try cache lookup for struct/class types that are declarations
        if entry.tag() == gimli::DW_TAG_structure_type
            || entry.tag() == gimli::DW_TAG_class_type
            || entry.tag() == gimli::DW_TAG_union_type
        {
            if let Some(name_attr) = entry.attr(gimli::DW_AT_name) {
                if let Ok(name) = self.dwarf.attr_string(unit, name_attr.value()) {
                    if let Ok(name_str) = name.to_string_lossy() {
                        if let Some(&(_, _, Some(size))) = self.type_defs.get(name_str.as_ref()) {
                            return Some(size);
                        }
                    }
                }
            }
        }

        None
    }

    fn get_type_name(
        &self,
        unit: &gimli::Unit<R>,
        type_offset: UnitOffset<R::Offset>,
    ) -> Option<String> {
        let mut cursor = unit.entries_at_offset(type_offset).ok()?;
        cursor.next_entry().ok()?;
        let entry = cursor.current()?;

        // Try to get name directly
        if let Some(attr) = entry.attr(gimli::DW_AT_name) {
            if let Ok(name) = self.dwarf.attr_string(unit, attr.value()) {
                if let Ok(s) = name.to_string_lossy() {
                    return Some(s.into_owned());
                }
            }
        }

        // For pointer types
        if entry.tag() == gimli::DW_TAG_pointer_type {
            if let Some(type_attr) = entry.attr(gimli::DW_AT_type) {
                if let gimli::AttributeValue::UnitRef(inner_offset) = type_attr.value() {
                    if let Some(base_name) = self.get_type_name(unit, inner_offset) {
                        return Some(format!("{} *", base_name));
                    }
                }
            }
            return Some("void *".to_string());
        }

        // For const types
        if entry.tag() == gimli::DW_TAG_const_type {
            if let Some(type_attr) = entry.attr(gimli::DW_AT_type) {
                if let gimli::AttributeValue::UnitRef(inner_offset) = type_attr.value() {
                    if let Some(base_name) = self.get_type_name(unit, inner_offset) {
                        return Some(format!("const {}", base_name));
                    }
                }
            }
        }

        // For typedef
        if entry.tag() == gimli::DW_TAG_typedef {
            if let Some(attr) = entry.attr(gimli::DW_AT_name) {
                if let Ok(name) = self.dwarf.attr_string(unit, attr.value()) {
                    if let Ok(s) = name.to_string_lossy() {
                        return Some(s.into_owned());
                    }
                }
            }
        }

        None
    }

    /// Check if type is a struct and return info for extracting members
    fn get_struct_info(
        &self,
        unit: &gimli::Unit<R>,
        type_offset: UnitOffset<R::Offset>,
    ) -> Option<(usize, UnitOffset<R::Offset>)> {
        let mut cursor = unit.entries_at_offset(type_offset).ok()?;
        cursor.next_entry().ok()?;
        let entry = cursor.current()?;

        if entry.tag() == gimli::DW_TAG_structure_type
            || entry.tag() == gimli::DW_TAG_class_type
            || entry.tag() == gimli::DW_TAG_union_type
        {
            let is_declaration = entry
                .attr(gimli::DW_AT_declaration)
                .map(|a| matches!(a.value(), gimli::AttributeValue::Flag(true)))
                .unwrap_or(false);

            if is_declaration {
                // Look up definition in cache by name
                if let Some(name_attr) = entry.attr(gimli::DW_AT_name) {
                    if let Ok(name) = self.dwarf.attr_string(unit, name_attr.value()) {
                        if let Ok(name_str) = name.to_string_lossy() {
                            if let Some(&(unit_idx, offset, _)) =
                                self.type_defs.get(name_str.as_ref())
                            {
                                let offset = UnitOffset(R::Offset::from_u64(offset).ok()?);
                                return Some((unit_idx, offset));
                            }
                        }
                    }
                }
                return None;
            }

            // Find which unit index this is
            for (idx, u) in self.units.iter().enumerate() {
                if std::ptr::eq(u, unit) {
                    return Some((idx, type_offset));
                }
            }
        }

        // Follow const/volatile/typedef
        if entry.tag() == gimli::DW_TAG_const_type
            || entry.tag() == gimli::DW_TAG_volatile_type
            || entry.tag() == gimli::DW_TAG_typedef
            || entry.tag() == gimli::DW_TAG_restrict_type
            || entry.tag() == gimli::DW_TAG_atomic_type
        {
            if let Some(type_attr) = entry.attr(gimli::DW_AT_type) {
                if let gimli::AttributeValue::UnitRef(inner_offset) = type_attr.value() {
                    return self.get_struct_info(unit, inner_offset);
                }
            }
        }

        None
    }

    fn extract_members(
        &self,
        unit_idx: usize,
        struct_offset: UnitOffset<R::Offset>,
        depth: usize,
    ) -> Result<Vec<FrameMember>> {
        const MAX_DEPTH: usize = 10;
        if depth > MAX_DEPTH {
            return Ok(Vec::new());
        }

        let unit = &self.units[unit_idx];
        let mut members = Vec::new();

        let mut cursor = unit.entries_at_offset(struct_offset)?;
        cursor.next_entry()?;

        while let Some(entry) = cursor.next_dfs()? {
            if entry.depth <= 0 {
                break;
            }
            if entry.depth != 1 {
                continue;
            }

            let is_inheritance = entry.tag() == gimli::DW_TAG_inheritance;
            let is_member = entry.tag() == gimli::DW_TAG_member;

            if !is_member && !is_inheritance {
                continue;
            }

            let name = if is_inheritance {
                "<base>".to_string()
            } else if let Some(attr) = entry.attr(gimli::DW_AT_name) {
                self.dwarf
                    .attr_string(unit, attr.value())
                    .ok()
                    .and_then(|s| s.to_string_lossy().ok().map(|c| c.into_owned()))
                    .unwrap_or_else(|| "<anonymous>".to_string())
            } else {
                "<anonymous>".to_string()
            };

            let type_name = entry
                .attr(gimli::DW_AT_type)
                .and_then(|attr| {
                    if let gimli::AttributeValue::UnitRef(offset) = attr.value() {
                        self.get_type_name(unit, offset)
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "<unknown>".to_string());

            let member_offset = entry
                .attr(gimli::DW_AT_data_member_location)
                .and_then(|a| a.udata_value())
                .unwrap_or(0);

            let is_artificial = entry
                .attr(gimli::DW_AT_artificial)
                .map(|a| matches!(a.value(), gimli::AttributeValue::Flag(true)))
                .unwrap_or(false);

            let (size, nested_members) = if let Some(attr) = entry.attr(gimli::DW_AT_type) {
                if let gimli::AttributeValue::UnitRef(type_offset) = attr.value() {
                    let size = self.get_type_size(unit, type_offset);

                    let nested = if let Some((nested_unit_idx, nested_offset)) =
                        self.get_struct_info(unit, type_offset)
                    {
                        self.extract_members(nested_unit_idx, nested_offset, depth + 1)
                            .ok()
                            .filter(|v| !v.is_empty())
                    } else {
                        None
                    };

                    (size, nested)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            members.push(FrameMember {
                name,
                type_name,
                offset: member_offset,
                size,
                size_estimated: false,
                is_artificial,
                nested_members,
            });
        }

        members.sort_by_key(|m| m.offset);
        Self::estimate_unknown_sizes(&mut members);

        Ok(members)
    }

    fn estimate_unknown_sizes(members: &mut [FrameMember]) {
        for i in 0..members.len() {
            if members[i].size.is_none() {
                let next_offset = members[i + 1..]
                    .iter()
                    .find(|m| m.offset > members[i].offset)
                    .map(|m| m.offset);

                if let Some(next_off) = next_offset {
                    members[i].size = Some(next_off - members[i].offset);
                    members[i].size_estimated = true;
                }
            }

            if let Some(ref mut nested) = members[i].nested_members {
                Self::estimate_unknown_sizes(nested);
            }
        }
    }

    fn extract_frame_info(&self, frame_type_name: &str) -> Result<Option<FrameInfo>> {
        for (unit_idx, unit) in self.units.iter().enumerate() {
            let mut entries = unit.entries();
            while let Some(entry) = entries.next_dfs()? {
                if entry.tag() == gimli::DW_TAG_structure_type {
                    if let Some(name_attr) = entry.attr(gimli::DW_AT_name) {
                        if let Ok(name) = self.dwarf.attr_string(unit, name_attr.value()) {
                            if let Ok(name_str) = name.to_string_lossy() {
                                if name_str == frame_type_name {
                                    let size = entry
                                        .attr(gimli::DW_AT_byte_size)
                                        .and_then(|a| a.udata_value())
                                        .unwrap_or(0);

                                    let members =
                                        self.extract_members(unit_idx, entry.offset(), 0)?;

                                    return Ok(Some(FrameInfo {
                                        type_name: name_str.into_owned(),
                                        size,
                                        members,
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    fn get_source_location(
        &self,
        unit: &gimli::Unit<R>,
        entry: &gimli::DebuggingInformationEntry<R>,
    ) -> Option<(String, u64)> {
        let file_attr = entry.attr(gimli::DW_AT_decl_file)?;
        let line = entry
            .attr(gimli::DW_AT_decl_line)
            .and_then(|a| a.udata_value())?;

        // Get file index
        let file_idx = match file_attr.value() {
            gimli::AttributeValue::FileIndex(idx) => idx,
            _ => return None,
        };

        let line_program = unit.line_program.as_ref()?;
        let header = line_program.header();

        let file_entry = header.file(file_idx)?;
        let file = if let Some(dir) = file_entry.directory(header) {
            let dir_str = self.dwarf.attr_string(unit, dir).ok()?;
            let file_str = self.dwarf.attr_string(unit, file_entry.path_name()).ok()?;
            format!(
                "{}/{}",
                dir_str.to_string_lossy().ok()?,
                file_str.to_string_lossy().ok()?
            )
        } else {
            let file_str = self.dwarf.attr_string(unit, file_entry.path_name()).ok()?;
            file_str.to_string_lossy().ok()?.into_owned()
        };

        Some((file, line))
    }

    fn enrich_coroutines(&self, coroutines: &mut HashMap<String, Coroutine>) -> Result<()> {
        // Build address -> function info map
        let mut addr_info: HashMap<u64, DwarfFunctionInfo> = HashMap::new();

        for unit in &self.units {
            // Build DIE cache for specification lookups
            let mut die_cache: HashMap<UnitOffset<R::Offset>, DwarfFunctionInfo> = HashMap::new();
            let mut entries = unit.entries();
            while let Some(entry) = entries.next_dfs()? {
                if entry.tag() == gimli::DW_TAG_subprogram {
                    let offset = entry.offset();
                    let name = entry.attr(gimli::DW_AT_name).and_then(|attr| {
                        self.dwarf
                            .attr_string(unit, attr.value())
                            .ok()
                            .and_then(|s| s.to_string_lossy().ok().map(|cow| cow.into_owned()))
                    });

                    let (file, line) = self.get_source_location(unit, entry).unzip();

                    die_cache.insert(offset, DwarfFunctionInfo { name, file, line });
                }
            }

            // Second pass: collect address -> info
            let mut entries = unit.entries();
            while let Some(entry) = entries.next_dfs()? {
                if entry.tag() == gimli::DW_TAG_subprogram {
                    let address = entry
                        .attr(gimli::DW_AT_low_pc)
                        .and_then(|a| match a.value() {
                            gimli::AttributeValue::Addr(addr) => Some(addr),
                            gimli::AttributeValue::DebugAddrIndex(idx) => {
                                self.dwarf.address(unit, idx).ok()
                            }
                            _ => None,
                        });

                    if let Some(addr) = address {
                        let mut info = DwarfFunctionInfo {
                            name: None,
                            file: None,
                            line: None,
                        };

                        // Get name
                        if let Some(attr) = entry.attr(gimli::DW_AT_name) {
                            if let Ok(name) = self.dwarf.attr_string(unit, attr.value()) {
                                if let Ok(s) = name.to_string_lossy() {
                                    info.name = Some(s.into_owned());
                                }
                            }
                        }

                        // Get location
                        if let Some((file, line)) = self.get_source_location(unit, entry) {
                            info.file = Some(file);
                            info.line = Some(line);
                        }

                        // Follow specification for lambdas
                        if info.file.is_none() || info.line.is_none() {
                            if let Some(spec_attr) = entry.attr(gimli::DW_AT_specification) {
                                if let gimli::AttributeValue::UnitRef(spec_offset) =
                                    spec_attr.value()
                                {
                                    if let Some(spec_info) = die_cache.get(&spec_offset) {
                                        if info.name.is_none() {
                                            info.name = spec_info.name.clone();
                                        }
                                        if info.file.is_none() {
                                            info.file = spec_info.file.clone();
                                        }
                                        if info.line.is_none() {
                                            info.line = spec_info.line;
                                        }
                                    }
                                }
                            }
                        }

                        addr_info.insert(addr, info);
                    }
                }
            }
        }

        // Enrich coroutines
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

            // Extract frame info
            let frame_type_name = coro.frame_type_name();
            if let Ok(Some(frame)) = self.extract_frame_info(&frame_type_name) {
                coro.frame = Some(frame);
            }
        }

        Ok(())
    }
}

/// Find coroutines by scanning symbol table
fn find_coroutines<'a>(object: &'a object::File) -> HashMap<String, Coroutine> {
    let mut coroutines: HashMap<String, Coroutine> = HashMap::new();

    for symbol in object.symbols() {
        let Ok(name) = symbol.name() else { continue };
        let address = symbol.address();
        let size = symbol.size();

        if name.ends_with(".resume") {
            let base = name.strip_suffix(".resume").unwrap();
            let coro = coroutines
                .entry(base.to_string())
                .or_insert_with(|| Coroutine::new(base.to_string()));
            coro.resume = Some(FunctionInfo {
                address,
                size,
                demangled_name: None,
                file: None,
                line: None,
            });
        } else if name.ends_with(".destroy") {
            let base = name.strip_suffix(".destroy").unwrap();
            let coro = coroutines
                .entry(base.to_string())
                .or_insert_with(|| Coroutine::new(base.to_string()));
            coro.destroy = Some(FunctionInfo {
                address,
                size,
                demangled_name: None,
                file: None,
                line: None,
            });
        }
    }

    // Find ramp functions
    for symbol in object.symbols() {
        let Ok(name) = symbol.name() else { continue };
        if coroutines.contains_key(name) {
            let coro = coroutines.get_mut(name).unwrap();
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

/// Load binary and extract coroutines with DWARF info
fn load_coroutines(bin: &PathBuf) -> Result<Vec<Coroutine>> {
    let file =
        File::open(bin).with_context(|| format!("Failed to open binary: {}", bin.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("Failed to memory-map binary: {}", bin.display()))?;
    let object = object::File::parse(&*mmap)
        .with_context(|| format!("Failed to parse binary: {}", bin.display()))?;

    let mut coroutines = find_coroutines(&object);

    // Load DWARF sections
    let load_section = |id: gimli::SectionId| -> Result<Cow<'_, [u8]>> {
        Ok(object
            .section_by_name(id.name())
            .map(|s| s.data().unwrap_or(&[]))
            .map(Cow::Borrowed)
            .unwrap_or(Cow::Borrowed(&[])))
    };

    let dwarf_sections = DwarfSections::load(load_section)?;
    let dwarf = dwarf_sections.borrow(|section| EndianSlice::new(section, LittleEndian));

    let analyzer = DwarfAnalyzer::new(&dwarf)?;
    analyzer.enrich_coroutines(&mut coroutines)?;

    let mut result: Vec<_> = coroutines.into_values().collect();
    result.sort_by(|a, b| a.display_name().cmp(b.display_name()));

    Ok(result)
}

// ============================================================================
// Commands
// ============================================================================

#[derive(Tabled)]
struct CoroutineRow {
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Location")]
    location: String,
    #[tabled(rename = "Frame Size (bytes)")]
    frame_size: String,
}

fn cmd_list(coroutines: &[Coroutine]) {
    let rows: Vec<CoroutineRow> = coroutines
        .iter()
        .map(|c| CoroutineRow {
            name: c.display_name().to_string(),
            location: c.location().unwrap_or_default(),
            frame_size: c
                .frame_size()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "?".to_string()),
        })
        .collect();

    let table = Table::new(&rows).with(Style::sharp()).to_string();
    println!("{}", table);
}

fn cmd_details(coroutines: &[Coroutine], name: &str) -> Result<()> {
    let matches: Vec<_> = coroutines
        .iter()
        .filter(|c| {
            c.display_name() == name
                || c.linkage_name == name
                || c.location().as_deref() == Some(name)
        })
        .collect();

    if matches.is_empty() {
        eprintln!("{} No coroutine found matching '{}'", "Error:".red(), name);
        eprintln!("\nAvailable coroutines:");
        for c in coroutines {
            let loc = c
                .location()
                .map(|l| format!(" ({})", l))
                .unwrap_or_default();
            eprintln!("  - {}{}", c.display_name(), loc);
        }
        bail!("Coroutine not found");
    }

    if matches.len() > 1 {
        eprintln!(
            "{} Multiple coroutines match '{}'. Please use location:",
            "Error:".red(),
            name
        );
        for c in &matches {
            let loc = c
                .location()
                .map(|l| format!(" ({})", l))
                .unwrap_or_default();
            eprintln!("  - {}{}", c.display_name(), loc);
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
        coro.location().unwrap_or_else(|| "?".to_string()).cyan()
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

    // Print frame structure
    if let Some(ref frame) = coro.frame {
        println!();
        println!("{}", "## Frame Structure".bold());
        println!();
        println!(
            "{} {} (total size: {} bytes)",
            "struct".purple(),
            frame.type_name.cyan(),
            frame.size
        );
        println!();

        let mut rows: Vec<FrameTableRow> = Vec::new();
        collect_frame_rows(&frame.members, 0, 0, &mut rows);

        let table = Table::new(&rows).with(Style::sharp()).to_string();
        println!("{}", table);
    } else {
        println!();
        println!("{} Frame structure not found in DWARF", "Warning:".yellow());
    }

    Ok(())
}

#[derive(Tabled)]
struct FrameTableRow {
    #[tabled(rename = "Field")]
    field: String,
    #[tabled(rename = "Offset")]
    offset: String,
    #[tabled(rename = "Size")]
    size: String,
}

fn collect_frame_rows(
    members: &[FrameMember],
    indent_level: usize,
    base_offset: u64,
    rows: &mut Vec<FrameTableRow>,
) {
    let indent = "  ".repeat(indent_level);

    for member in members {
        let size_str = match (member.size, member.size_estimated) {
            (Some(s), true) => format!("~{}", s),
            (Some(s), false) => s.to_string(),
            (None, _) => "?".to_string(),
        };

        let artificial_marker = if member.is_artificial { " [a]" } else { "" };
        let field = format!(
            "{}{} {}{}",
            indent, member.type_name, member.name, artificial_marker
        );

        let absolute_offset = base_offset + member.offset;

        rows.push(FrameTableRow {
            field,
            offset: format!("0x{:x}", absolute_offset),
            size: size_str,
        });

        if let Some(ref nested) = member.nested_members {
            collect_frame_rows(nested, indent_level + 1, absolute_offset, rows);
        }
    }
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
