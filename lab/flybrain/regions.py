"""Anatomical regions (neuropils) of the Drosophila brain.

Abbreviations follow the systematic nomenclature of Ito et al. (2014),
Curr Biol 24:758 -- the same one FlyWire and the hemibrain use for cell-type
names.  A central-brain cell type is named after the neuropil it mainly
arborises in (AVLP429 -> AVLP, SMP012 -> SMP), so the type prefix is a usable
region label.  Neurons whose type carries no neuropil prefix get filled in by
nearest neighbours in soma space; see data.assign_regions.
"""

# abbreviation -> (russian name, super-region)
NEUROPILS = {
    # --- optic lobe ---
    "LA": ("ламина", "optic"),
    "ME": ("медулла", "optic"),
    "AME": ("доп. медулла", "optic"),
    "LO": ("лобула", "optic"),
    "LOP": ("пластина лобулы", "optic"),
    # --- mushroom body ---
    "MB": ("грибовидное тело", "mushroom body"),
    "CA": ("чашечка ГТ", "mushroom body"),
    "PED": ("цветоножка ГТ", "mushroom body"),
    # --- central complex ---
    "EB": ("эллипсоидное тело", "central complex"),
    "FB": ("веерообразное тело", "central complex"),
    "PB": ("протоцеребральный мостик", "central complex"),
    "NO": ("нодули", "central complex"),
    "AB": ("асимметричное тело", "central complex"),
    "BU": ("булава", "central complex"),
    "GA": ("гал", "central complex"),
    "LAL": ("латеральное дополнительное тело", "central complex"),
    # --- antennal lobe / olfactory ---
    "AL": ("антеннальная долька", "olfactory"),
    "LH": ("латеральный рог", "olfactory"),
    # --- superior protocerebrum ---
    "SLP": ("верхний латеральный протоцеребрум", "superior"),
    "SIP": ("верхний промежуточный протоцеребрум", "superior"),
    "SMP": ("верхний медиальный протоцеребрум", "superior"),
    # --- inferior protocerebrum ---
    "CRE": ("крепис", "inferior"),
    "ROB": ("роб", "inferior"),
    "RUB": ("рубус", "inferior"),
    "SCL": ("верхняя клампа", "inferior"),
    "ICL": ("нижняя клампа", "inferior"),
    "IB": ("нижний мост", "inferior"),
    "ATL": ("дополнительная доля", "inferior"),
    # --- ventrolateral protocerebrum ---
    "AOTU": ("передний зрительный бугорок", "ventrolateral"),
    "AVLP": ("передний вентролатеральный протоцеребрум", "ventrolateral"),
    "PVLP": ("задний вентролатеральный протоцеребрум", "ventrolateral"),
    "PLP": ("задний латеральный протоцеребрум", "ventrolateral"),
    "WED": ("клин", "ventrolateral"),
    # --- ventromedial ---
    "VES": ("вест", "ventromedial"),
    "EPA": ("эпаулетка", "ventromedial"),
    "GOR": ("горже", "ventromedial"),
    "SPS": ("верхний задний щиток", "ventromedial"),
    "IPS": ("нижний задний щиток", "ventromedial"),
    "PS": ("задний щиток", "ventromedial"),
    # --- periesophageal ---
    "CAN": ("каникулюс", "periesophageal"),
    "PRW": ("ростральный клин", "periesophageal"),
    "FLA": ("фланец", "periesophageal"),
    # --- gnathal / subesophageal zone ---
    "GNG": ("гнатальные ганглии", "gnathal (SEZ)"),
    "SAD": ("субэзофагеальный дорсальный", "gnathal (SEZ)"),
    "AMMC": ("антеннальный механосенсорный центр", "gnathal (SEZ)"),
    # --- outside the brain proper ---
    "VNC": ("грудной ганглий (ВНЦ)", "outside brain"),
    "NERVE": ("нерв / периферия", "outside brain"),
}

# Prefixes matched longest-first, so AOTU beats AO and LOP beats LO.
_PREFIXES = sorted(NEUROPILS, key=len, reverse=True)

# Named cell families whose name does NOT start with their neuropil.  Checked
# before the plain prefixes and case-sensitively, because FlyWire type names are
# consistent about case (MeTu, vDelta7, hDeltaK).
SPECIAL_FAMILIES = {
    # central complex: ellipsoid body
    "ER": "EB",
    "ExR": "EB",
    "EL": "EB",
    "EPG": "EB",
    "PEG": "EB",
    "PEN": "EB",
    # central complex: fan-shaped body
    "PFN": "FB",
    "PFL": "FB",
    "PFR": "FB",
    "PFG": "FB",
    "FC": "FB",
    "FS": "FB",
    "FR": "FB",
    "FX": "FB",
    "vDelta": "FB",
    "hDelta": "FB",
    # central complex: protocerebral bridge and noduli
    "Delta7": "PB",
    "SpsP": "PB",
    "IbSpsP": "PB",
    "LNO": "NO",
    "GLNO": "NO",
    "LCNO": "NO",
    # bulb and anterior optic tubercle
    "TuBu": "BU",
    "TuTu": "AOTU",
    "MeTu": "AOTU",
    "AOTU": "AOTU",
    # mushroom body
    "KC": "MB",
    "MBON": "MB",
    "MBIN": "MB",
    "APL": "MB",
    "PAM": "MB",
    "PPL1": "MB",
    "PPL2": "MB",
    # lateral horn
    "LHAV": "LH",
    "LHPV": "LH",
    "LHCENT": "LH",
    "LHML": "LH",
}

_SPECIAL_SORTED = sorted(SPECIAL_FAMILIES.items(), key=lambda kv: -len(kv[0]))

# cell_class values that pin a neuron to a region regardless of its type name.
CLASS_TO_REGION = {
    "Kenyon_Cell": "MB",
    "MBON": "MB",
    "MBIN": "MB",
    "DAN": "MB",
    "APL": "MB",
    "ALPN": "AL",
    "ALLN": "AL",
    "ALIN": "AL",
    "ALON": "AL",
    "olfactory": "AL",
    "hygrosensory": "AL",
    "thermosensory": "AL",
    "LHCENT": "LH",
    "LHLN": "LH",
    "LHON": "LH",
    "ER": "EB",
    "EPG": "EB",
    "PEN": "EB",
    "PEG": "EB",
    "ring neuron": "EB",
    # Gustatory receptor axons terminate in the gnathal ganglia, which is what
    # "subesophageal zone" names; keeping a separate SEZ label would split one
    # region in two.
    "gustatory": "GNG",
    "mechanosensory": "AMMC",
    "bristle": "GNG",
    "visual": "ME",
    "ocellar": "PLP",
    "optic_lobes": "ME",
    "motor": "NERVE",
    "endocrine": "NERVE",
    "descending": "GNG",
    "ascending": "VNC",
    "visual_projection": "LO",
    "visual_centrifugal": "LO",
    "AN": "VNC",
    "DN": "GNG",
}

SUPER_TO_REGION = {
    "optic": "ME",
    "sensory": "NERVE",
    "motor": "NERVE",
    "endocrine": "NERVE",
    "descending": "GNG",
    "ascending": "VNC",
    "visual_projection": "LO",
    "visual_centrifugal": "LO",
}

# Optic-lobe intrinsic families, named after the neuropil they live in.
OPTIC_FAMILIES = {
    "L": "LA",
    "Lawf": "LA",
    "C2": "LA",
    "C3": "LA",
    "R1": "LA",
    "R7": "ME",
    "R8": "ME",
    "Mi": "ME",
    "Tm": "ME",
    "TmY": "ME",
    "Dm": "ME",
    "Pm": "ME",
    "Mt": "ME",
    "MeTu": "ME",
    "MeMe": "ME",
    "Me": "ME",
    "T1": "ME",
    "T2": "ME",
    "T3": "ME",
    "T4": "ME",
    "T5": "LOP",
    "Y": "LO",
    "Tlp": "LO",
    "Li": "LO",
    "Lo": "LO",
    "LC": "LO",
    "LT": "LO",
    "LPLC": "LO",
    "LLPC": "LO",
    "LPC": "LO",
    "LPi": "LOP",
    "LPT": "LOP",
    "H1": "LOP",
    "H2": "LOP",
    "HS": "LOP",
    "VS": "LOP",
    "CH": "LOP",
    "Am": "AME",
}

_OPTIC_SORTED = sorted(OPTIC_FAMILIES.items(), key=lambda kv: -len(kv[0]))


def _clean(cell_type):
    if cell_type is None:
        return None
    name = str(cell_type).strip()
    if not name or name.lower() in ("nan", "none"):
        return None
    # CB0924 and friends are placeholders for unnamed central-brain cells.
    if name.upper().startswith("CB") and name[2:3].isdigit():
        return None
    return name


def region_from_type(cell_type):
    """Neuropil abbreviation implied by a FlyWire cell-type name, or None.

    Only the strong signals: a named cell family (ER4d, MeTu1, KCab) or a
    neuropil abbreviation used as a prefix (AVLP429, SMP012).  Single-letter
    optic families like L2 or Tm9 are deliberately left to
    :func:`region_from_optic_family`, because sensory type names such as the
    labellar bristle GRN ``LB3c`` would otherwise be read as lamina cells.
    """
    name = _clean(cell_type)
    if name is None:
        return None
    for fam, reg in _SPECIAL_SORTED:
        if name.startswith(fam):
            return reg
    upper = name.upper()
    for prefix in _PREFIXES:
        if upper.startswith(prefix):
            return prefix
    return None


def region_from_optic_family(cell_type):
    """Neuropil for an optic-lobe intrinsic family (L2, Tm9, T4a, Dm8, ...).

    Applied only to neurons the annotation already calls optic, so it cannot
    swallow unrelated names that happen to start with the same letter.
    """
    name = _clean(cell_type)
    if name is None:
        return None
    for fam, reg in _OPTIC_SORTED:
        if name.startswith(fam):
            return reg
    return None


def region_label(abbr):
    """'AVLP' -> 'AVLP — передний вентролатеральный протоцеребрум'."""
    if abbr in NEUROPILS:
        return "{} — {}".format(abbr, NEUROPILS[abbr][0])
    return str(abbr)


def super_region(abbr):
    return NEUROPILS.get(abbr, ("", "other"))[1]
