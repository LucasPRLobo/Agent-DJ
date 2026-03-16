"""DJ-friendly genre taxonomy mapping from Discogs-400 labels.

Maps the 400 Discogs genre/style labels to a simplified, DJ-relevant
taxonomy organized by broad categories and sub-genres.

A track can belong to multiple DJ genres with different weights,
derived from the Discogs-400 classifier output.
"""

from dataclasses import dataclass, field

# DJ Genre Taxonomy
# Each key is a DJ-friendly genre label. Values are lists of Discogs-400 labels
# that map to it. When a track's Discogs predictions include these labels,
# the confidence scores are aggregated into the DJ genre.

TAXONOMY: dict[str, list[str]] = {
    # === ELECTRONIC ===
    "House": [
        "Electronic---House", "Electronic---Deep House", "Electronic---Tech House",
        "Electronic---Acid House", "Electronic---Garage House", "Electronic---Electro House",
        "Electronic---Progressive House", "Electronic---Euro House", "Electronic---Italo House",
        "Electronic---Ghetto House", "Electronic---Hard House", "Electronic---Tribal House",
        "Electronic---Tropical House", "Electronic---Hip-House",
    ],
    "Techno": [
        "Electronic---Techno", "Electronic---Minimal Techno", "Electronic---Deep Techno",
        "Electronic---Hard Techno", "Electronic---Dub Techno", "Electronic---Schranz",
        "Electronic---Minimal",
    ],
    "Disco / Nu-Disco": [
        "Electronic---Disco", "Electronic---Nu-Disco", "Electronic---Italo-Disco",
        "Electronic---Euro-Disco", "Electronic---Disco Polo", "Funk / Soul---Disco",
    ],
    "Trance": [
        "Electronic---Trance", "Electronic---Goa Trance", "Electronic---Progressive Trance",
        "Electronic---Psy-Trance", "Electronic---Hard Trance", "Electronic---Tech Trance",
    ],
    "Drum & Bass / Jungle": [
        "Electronic---Drum n Bass", "Electronic---Jungle", "Electronic---Halftime",
        "Electronic---Breakcore",
    ],
    "Breakbeat": [
        "Electronic---Breakbeat", "Electronic---Breaks", "Electronic---Progressive Breaks",
        "Electronic---Big Beat",
    ],
    "Dubstep / Bass": [
        "Electronic---Dubstep", "Electronic---Bassline", "Electronic---Bleep",
        "Electronic---Grime", "Hip Hop---Bass Music",
    ],
    "UK Garage / 2-Step": [
        "Electronic---UK Garage", "Electronic---Speed Garage", "Electronic---Beatdown",
    ],
    "Ambient / Downtempo": [
        "Electronic---Ambient", "Electronic---Downtempo", "Electronic---Dark Ambient",
        "Electronic---Chillwave", "Electronic---New Age", "Electronic---Trip Hop",
        "Electronic---IDM", "Electronic---Drone",
    ],
    "Electro": [
        "Electronic---Electro", "Electronic---Electroclash", "Electronic---EBM",
    ],
    "Synth / Wave": [
        "Electronic---Synth-pop", "Electronic---Synthwave", "Electronic---Darkwave",
        "Electronic---New Wave", "Electronic---New Beat",
    ],
    "Hardcore / Hardstyle": [
        "Electronic---Hardcore", "Electronic---Hardstyle", "Electronic---Gabber",
        "Electronic---Happy Hardcore", "Electronic---Speedcore", "Electronic---Jumpstyle",
        "Electronic---Hands Up", "Electronic---Hi NRG", "Electronic---Makina",
    ],
    "Experimental Electronic": [
        "Electronic---Experimental", "Electronic---Glitch", "Electronic---Abstract",
        "Electronic---Noise", "Electronic---Sound Collage", "Electronic---Leftfield",
        "Electronic---Musique Concrète", "Electronic---Illbient", "Electronic---Vaporwave",
        "Electronic---Power Electronics", "Electronic---Rhythmic Noise",
    ],

    # === URBAN ===
    "RnB / Neo-Soul": [
        "Funk / Soul---Contemporary R&B", "Funk / Soul---Neo Soul",
        "Funk / Soul---New Jack Swing", "Funk / Soul---Swingbeat",
        "Funk / Soul---UK Street Soul", "Hip Hop---RnB/Swing",
    ],
    "Hip-Hop": [
        "Hip Hop---Boom Bap", "Hip Hop---Gangsta", "Hip Hop---Conscious",
        "Hip Hop---Hardcore Hip-Hop", "Hip Hop---Thug Rap", "Hip Hop---Pop Rap",
        "Hip Hop---Instrumental", "Hip Hop---Jazzy Hip-Hop", "Hip Hop---Horrorcore",
        "Hip Hop---Britcore", "Hip Hop---Cloud Rap",
    ],
    "Trap": [
        "Hip Hop---Trap", "Hip Hop---Crunk", "Hip Hop---Screw",
        "Hip Hop---Bounce", "Hip Hop---Miami Bass",
    ],
    "Grime": [
        "Hip Hop---Grime", "Electronic---Grime",
    ],
    "G-Funk": [
        "Hip Hop---G-Funk",
    ],
    "DJ / Turntablism": [
        "Hip Hop---Cut-up/DJ", "Hip Hop---DJ Battle Tool", "Hip Hop---Turntablism",
    ],

    # === SOUL / FUNK ===
    "Soul": [
        "Funk / Soul---Soul", "Funk / Soul---Rhythm & Blues", "Funk / Soul---Gospel",
        "Blues---Rhythm & Blues",
    ],
    "Funk": [
        "Funk / Soul---Funk", "Funk / Soul---P.Funk", "Funk / Soul---Free Funk",
        "Funk / Soul---Boogie", "Funk / Soul---Psychedelic",
    ],
    "Afrobeat": [
        "Funk / Soul---Afrobeat", "Jazz---Afrobeat",
    ],

    # === JAZZ ===
    "Jazz": [
        "Jazz---Bop", "Jazz---Hard Bop", "Jazz---Cool Jazz", "Jazz---Modal",
        "Jazz---Post Bop", "Jazz---Swing", "Jazz---Big Band", "Jazz---Dixieland",
        "Jazz---Free Jazz", "Jazz---Free Improvisation", "Jazz---Avant-garde Jazz",
        "Jazz---Ragtime", "Jazz---Space-Age", "Jazz---Gypsy Jazz",
    ],
    "Jazz Fusion": [
        "Jazz---Fusion", "Jazz---Jazz-Rock", "Jazz---Jazz-Funk",
        "Electronic---Future Jazz", "Electronic---Acid Jazz",
    ],
    "Smooth / Contemporary Jazz": [
        "Jazz---Smooth Jazz", "Jazz---Contemporary Jazz", "Jazz---Easy Listening",
        "Jazz---Soul-Jazz",
    ],
    "Bossa Nova / Latin Jazz": [
        "Jazz---Bossa Nova", "Jazz---Latin Jazz", "Jazz---Afro-Cuban Jazz",
        "Latin---Bossanova",
    ],

    # === LATIN ===
    "Salsa / Mambo": [
        "Latin---Salsa", "Latin---Mambo", "Latin---Cha-Cha", "Latin---Son",
        "Latin---Son Montuno", "Latin---Guaguancó", "Latin---Charanga",
        "Latin---Descarga", "Latin---Pachanga", "Latin---Guaracha",
        "Latin---Cubano", "Latin---Afro-Cuban", "Latin---Boogaloo",
        "Latin---Guajira", "Latin---Rumba",
    ],
    "Reggaeton": [
        "Latin---Reggaeton",
    ],
    "Brazilian": [
        "Latin---Samba", "Latin---MPB", "Latin---Forró", "Latin---Baião",
        "Latin---Batucada",
    ],
    "Cumbia": [
        "Latin---Cumbia", "Latin---Porro", "Latin---Vallenato",
    ],
    "Tango": [
        "Latin---Tango",
    ],
    "Other Latin": [
        "Latin---Bolero", "Latin---Beguine", "Latin---Compas",
        "Latin---Merengue", "Latin---Nueva Cancion", "Latin---Norteño",
        "Latin---Ranchera", "Latin---Tejano", "Latin---Mariachi",
    ],

    # === REGGAE ===
    "Reggae / Roots": [
        "Reggae---Reggae", "Reggae---Roots Reggae", "Reggae---Lovers Rock",
        "Reggae---Reggae-Pop",
    ],
    "Dub": [
        "Reggae---Dub", "Electronic---Dub",
    ],
    "Dancehall / Ragga": [
        "Reggae---Dancehall", "Reggae---Ragga", "Hip Hop---Ragga HipHop",
    ],
    "Ska / Rocksteady": [
        "Reggae---Ska", "Reggae---Rocksteady",
    ],
    "Soca / Calypso": [
        "Reggae---Soca", "Reggae---Calypso",
    ],

    # === POP ===
    "Pop": [
        "Pop---Ballad", "Pop---Vocal", "Pop---Europop", "Pop---Bubblegum",
        "Pop---Indie Pop", "Pop---Chanson", "Pop---Light Music",
        "Pop---Schlager", "Pop---City Pop", "Pop---J-pop", "Pop---K-pop",
        "Electronic---Dance-pop",
    ],
    "Bollywood": [
        "Pop---Bollywood",
    ],

    # === ROCK ===
    "Rock": [
        "Rock---Rock & Roll", "Rock---Classic Rock", "Rock---Hard Rock",
        "Rock---Prog Rock", "Rock---Psychedelic Rock", "Rock---Garage Rock",
        "Rock---Indie Rock", "Rock---Alternative Rock", "Rock---Punk",
        "Rock---Post-Punk", "Rock---New Wave", "Rock---Shoegaze",
        "Rock---Stoner Rock", "Rock---Grunge", "Rock---Emo",
    ],

    # === BLUES ===
    "Blues": [
        "Blues---Electric Blues", "Blues---Chicago Blues", "Blues---Delta Blues",
        "Blues---Modern Electric Blues", "Blues---Texas Blues", "Blues---Jump Blues",
        "Blues---Piano Blues", "Blues---Harmonica Blues", "Blues---Country Blues",
        "Blues---Louisiana Blues", "Blues---Boogie Woogie",
    ],

    # === WORLD ===
    "World / Folk": [
        "Folk, World, & Country---African", "Folk, World, & Country---Celtic",
        "Folk, World, & Country---Indian Classical", "Folk, World, & Country---Middle Eastern",
    ],
}

# Build reverse lookup: Discogs label → list of (DJ genre, weight)
_REVERSE_MAP: dict[str, list[str]] = {}
for dj_genre, discogs_labels in TAXONOMY.items():
    for dl in discogs_labels:
        _REVERSE_MAP.setdefault(dl, []).append(dj_genre)


@dataclass
class DJGenreProfile:
    """A track's genre profile mapped to the DJ taxonomy."""
    genres: dict[str, float] = field(default_factory=dict)  # DJ genre → aggregated score

    @property
    def primary_genre(self) -> str:
        """The highest-scoring DJ genre."""
        if not self.genres:
            return "Unknown"
        return max(self.genres, key=self.genres.get)

    @property
    def top_genres(self) -> list[tuple[str, float]]:
        """All genres sorted by score descending."""
        return sorted(self.genres.items(), key=lambda x: -x[1])

    def matches_any(self, target_genres: list[str], threshold: float = 0.05) -> bool:
        """Check if this track matches any of the target genres."""
        return any(self.genres.get(g, 0) >= threshold for g in target_genres)


def map_to_dj_genres(
    discogs_predictions: list[tuple[str, float]],
) -> DJGenreProfile:
    """Map Discogs-400 classifier output to DJ genre taxonomy.

    Args:
        discogs_predictions: List of (discogs_label, confidence) tuples
            from the genre classifier.

    Returns:
        DJGenreProfile with aggregated scores per DJ genre.
    """
    genre_scores: dict[str, float] = {}

    for discogs_label, confidence in discogs_predictions:
        dj_genres = _REVERSE_MAP.get(discogs_label, [])
        for dj_genre in dj_genres:
            genre_scores[dj_genre] = genre_scores.get(dj_genre, 0) + confidence

    # Filter out very low scores
    genre_scores = {k: round(v, 4) for k, v in genre_scores.items() if v >= 0.01}

    return DJGenreProfile(genres=genre_scores)


def get_all_dj_genres() -> list[str]:
    """Return all DJ genre names in the taxonomy."""
    return list(TAXONOMY.keys())


def get_genre_category(dj_genre: str) -> str:
    """Get the broad category for a DJ genre."""
    categories = {
        "Electronic": [
            "House", "Techno", "Disco / Nu-Disco", "Trance", "Drum & Bass / Jungle",
            "Breakbeat", "Dubstep / Bass", "UK Garage / 2-Step", "Ambient / Downtempo",
            "Electro", "Synth / Wave", "Hardcore / Hardstyle", "Experimental Electronic",
        ],
        "Urban": ["RnB / Neo-Soul", "Hip-Hop", "Trap", "Grime", "G-Funk", "DJ / Turntablism"],
        "Soul / Funk": ["Soul", "Funk", "Afrobeat"],
        "Jazz": ["Jazz", "Jazz Fusion", "Smooth / Contemporary Jazz", "Bossa Nova / Latin Jazz"],
        "Latin": ["Salsa / Mambo", "Reggaeton", "Brazilian", "Cumbia", "Tango", "Other Latin"],
        "Reggae": ["Reggae / Roots", "Dub", "Dancehall / Ragga", "Ska / Rocksteady", "Soca / Calypso"],
        "Pop": ["Pop", "Bollywood"],
        "Rock": ["Rock"],
        "Blues": ["Blues"],
        "World": ["World / Folk"],
    }
    for category, genres in categories.items():
        if dj_genre in genres:
            return category
    return "Other"
