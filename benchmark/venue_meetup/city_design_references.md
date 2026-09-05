# Venue Meetup city-design references

The authored districts borrow legible spatial rules from real cities. They
do not reproduce a copyrighted map or claim architectural fidelity; the
precedents are translated into deterministic geometry that can be spawned from
the packaged SimWorld asset catalogue.

## Station quarter: Barcelona's Eixample

Barcelona City Council describes the Cerdà plan as a strict grid of parallel
and perpendicular streets, interrupted by a few diagonal avenues, with planned
street widths of 20, 30, or 60 metres. Its institutional overview also
describes the Eixample as square blocks with chamfered corners and interior
courtyards.

Venue Meetup applies those rules at benchmark scale:

- four eight-sided blocks form a compact 2x2 grid;
- each corner has a 15 m chamfer;
- the 14 m carriageway plus two 3 m sidewalks creates a 20 m public corridor;
- Cross Street is a continuous orientation axis;
- the clock tower and station tower close the west and east terminal views;
- the market hall, hotel, and southeast tower provide north/south anchors.

Sources:

- [Barcelona City Council: The Cerdà Plan](https://ajuntament.barcelona.cat/eixample/sites/default/files/af_eixample_antigaesquerra_bx.pdf)
- [Barcelona City Council institutional repository: Eixample overview](https://bcnroc.ajuntament.barcelona.cat/jspui/bitstream/11703/91081/1/13005.pdf)
- [Barcelona City Council archive: dimensions and orientation of Eixample blocks](https://bcnroc.ajuntament.barcelona.cat/jspui/bitstream/11703/90651/1/4297.pdf)

## Riverside market: Amsterdam's canal district

UNESCO describes Amsterdam's canal ring through its canal-side streets,
radial connecting streets, regular building lines, narrow plots, bridges, and
repeated facade rhythm. It records typical regulated plot widths of roughly
6.2 to 8.5 metres and notes that corner buildings at radial streets received
stronger architectural accents.

Venue Meetup translates that hierarchy into:

- a central canal/barrier with a promenade on each bank;
- one outer avenue and one internal merchant lane per bank;
- twelve narrower blocks instead of six oversized superblocks;
- two explicit bridge crossings, with the southern crossing remaining a detour;
- vertically emphasized canal-house rows;
- civic, transit, market-hall, bridge-gate, hospital, and hotel landmarks at
  bridgeheads, terminal vistas, and major decisions.

Source:

- [UNESCO Urban Heritage Atlas: Amsterdam Canal Ring](https://whc.unesco.org/en/urban-heritage-atlas/Amsterdam)

## Rosebank grid: Johannesburg mixed-use transit district

The City of Johannesburg publishes a Rosebank Urban Development Framework for
Region B. The district's Gautrain station sits beneath Oxford Road between
Baker Street and Tyrwhitt Avenue, opposite The Zone, which gives the benchmark
a real, legible pair of primary axes. Current City and Johannesburg Development
Agency projects also emphasize mixed uses, reused buildings, green public
space, upgraded sidewalks and intersections, market activity, and continuous
pedestrian links around the station precinct.

Venue Meetup translates those ingredients rather than copying property lines:

- a 9x9 orthogonal grid spans approximately 684 m;
- Oxford Road is the north/south transit spine and Tyrwhitt the east/west high
  street;
- 7 transit-core, 53 mixed-use, 12 residential, 5 civic, and 4 garden blocks
  create a readable density gradient;
- 21 horizontal and 16 vertical mid-block alley axes add shortcuts without
  replacing the public street grid;
- 36 uniquely inspectable venues cover restaurants, cafes, shops, bookshops,
  bars, pubs, hotels, and tower lobbies;
- the Gautrain tower, market hall, hotel tower, arts centre, civic hall, and
  clock tower anchor relative positioning.

Sources:

- [City of Johannesburg Region B plans](https://joburg.org.za/documents_/Pages/Key%20Documents/other/policies/Development%20Planning%20%26%20Urban%20Management/Region%20B/Region-B-Plans.aspx)
- [Gautrain: Rosebank station information](https://www.gautrain.co.za/commuter/stationinfo?stationName=Rosebank)
- [City of Johannesburg: Nine Yards mixed-use precinct](https://joburg.org.za/media_/Newsroom/Pages/2026-News-Articles/Nine-Yards-Joburg%E2%80%99s-green-town-square-reaches-full-bloom.aspx)
- [Johannesburg Development Agency: Rosebank public transport facility phase 2](https://www.jda.org.za/wp-content/uploads/2025/06/20250617-PREPARATORY-WORK-COMMENCES-FOR-PHASE-2-OF-THE-ROSEBANK-PUBLIC-TRANSPORT-FACILITY-.pdf)

## Landmark placement and relative positioning

Transport for London recommends locating pedestrian wayfinding at journey
starts, key decision points, and landmark destinations, and notes that
recognizable 3D buildings help people interpret maps. The benchmark follows
that rule directly:

- agents spawn at interior street decisions rather than exposed map edges;
- major landmark assets are reserved and never reused as generic shells;
- major landmarks are vertically exaggerated while retaining measured
  collision footprints;
- public map labels use landmark names rather than generic building classes;
- coarse-map descriptions state each landmark's relation to the primary axes.

Sources:

- [Transport for London: maps and signs](https://tfl.gov.uk/info-for/boroughs-and-communities/maps-and-signs)
- [Transport for London Streetscape Guidance, section 11.12](https://content.tfl.gov.uk/streetscape-guidance-2022-revision-2.pdf)
