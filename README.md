# MappaTrakka

Turn a Strava bulk export into a glowing heatmap of the routes you travel most.
Every activity is drawn as a line and stacked on top of the others, so the roads
you use all the time burn white-hot while the one-off detours stay a faint red.

It works two ways:

- **In the browser**: drop your export onto the web page. Everything runs on your
  own machine (Python compiled to WebAssembly), so your GPS data is never uploaded.
- **On the command line**: point it at the export and get a 4000 × 4000 PNG.

Both use exactly the same code.

## Getting your data

On strava.com go to **Settings → My Account → Download or Delete Your Account →
Get Started → Request Your Archive**. Strava emails you a link to a zip. You can
use that zip as it is; there's no need to unzip it.

## Command line

Needs Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (on macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m mappatrakka path/to/export.zip
# -> output/heatmap.png
```

Some useful options:

```bash
python -m mappatrakka export.zip -o rides.png --include-types Ride "Gravel Ride"
python -m mappatrakka export.zip --size 6000 --colormap ice --glow 0
python -m mappatrakka export.zip --area 2            # second busiest area (holidays, a second home)
python -m mappatrakka export.zip --region all        # fit every activity, wherever it was
python -m mappatrakka export.zip --center 51.507 -0.128 --radius-km 10
python -m mappatrakka export.zip -c config.toml      # everything else lives in a config file
```

`python -m mappatrakka --help` lists the rest. Every setting, including the
colours, brightness curve, GPS clean-up thresholds and privacy zones, is
described in [config.example.toml](config.example.toml).

Parsed tracks are cached in `.mappatrakka-cache/`, so a second run with
different colours or a different area takes seconds instead of minutes.

### Privacy zones

A heatmap makes it obvious where you start and finish most rides. Before
sharing one, hide your home (or anywhere else) in your config:

```toml
[[privacy.zones]]
lat = 51.5074
lon = -0.1278
radius_m = 300
```

## Website

The `web/` folder is the in-browser version. To build and try it locally:

```bash
python scripts/build_web.py --serve      # then open http://127.0.0.1:8000
```

That writes a static site into `dist/`, which you can host anywhere. The
included GitHub Actions workflow (`.github/workflows/pages.yml`) publishes it to
GitHub Pages on every push to `main`, once Pages is set to deploy from
"GitHub Actions" in the repo settings.

The first visit downloads about 15 MB (Pyodide, numpy and Pillow), which the
browser then caches. Processing is slower than the command line: about a
quarter of a second per hour-long FIT activity (GPX and TCX are quicker), so
around 4 minutes for a thousand. In exchange, nothing leaves the visitor's device.
It takes the export zip, the unzipped folder, or a pile of loose `.fit`, `.gpx`
and `.tcx` files.

The "On a map" tab lays the result over OpenStreetMap (via Leaflet) so you can
see which streets you're looking at. The heatmap is already Web Mercator, the
same projection as the map tiles, so it lines up with just its corner
coordinates. Leaflet and the tiles are only fetched when that tab is opened, and
tile requests do tell OpenStreetMap roughly which area is being viewed.

No data handy? `python -m mappatrakka.demo` writes a fake export (a made-up
town, some holidays, virtual rides and a few broken files) to try things on,
and the website has a "Try with demo data" button.

## How it works

1. **Read the export.** `activities.csv` says which file belongs to which
   activity. Virtual rides, gym sessions, yoga and so on are dropped by type,
   by name ("Zwift", "indoor") and by what FIT files say about themselves
   (`sub_sport = virtual_activity`).
2. **Parse the tracks.** FIT (via [fitdecode](https://github.com/polyvertex/fitdecode)),
   GPX and TCX, gzipped or not. The format is detected from the file contents,
   not just its name. Damaged files keep whatever could be read before the damage.
3. **Clean them.** Invalid points are dropped, single-point GPS spikes removed,
   and lines are split wherever the recording jumps (lost signal, a train ride
   with the timer paused), so no straight lines get drawn across the map. GPX
   track segments and FIT timer stops are never joined up.
4. **Find home turf.** Each activity is reduced to its median point, the points
   are grouped by single-linkage clustering (30 km by default), and the group
   with the most activities wins. The map is sized to that group, ignoring the
   few activities that reach furthest out, so one 200 km epic doesn't zoom
   everything out. Routes from anywhere that cross the area are still drawn.
5. **Draw.** Web Mercator keeps local shapes true, so nothing looks stretched.
   Each activity is drawn once, anti-aliased, into its own layer capped at 1 per
   pixel, and the layers are summed. The total is "how many separate trips used
   this pixel", whatever the GPS recording rate and even for out-and-back routes.
6. **Colour.** The counts are mapped through a log curve, so a road used 200
   times doesn't drown out one used twice, then through a black → red → orange →
   yellow → white gradient, with a soft glow underneath the sharp lines.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests generate real FIT, GPX and TCX files, both plain and gzipped. They check
coordinate conversion, point order, segment handling and broken files, and that
holidays don't distort the map and repeated routes come out brighter. They also
check that output is identical between runs, and that a cached run and a parallel
run produce byte-identical images.

```
mappatrakka/
  export.py      activities.csv and filtering
  archive.py     reading the export zip
  parsers/       FIT, GPX, TCX (add a format with one register() call)
  processing.py  GPS clean-up
  projection.py  Web Mercator and map bounds
  region.py      finding the main area
  render.py      drawing and colouring
  pipeline.py    the whole run
  web.py         entry point for the browser version
web/             the website
scripts/         build_web.py
```

## Licence

MIT
