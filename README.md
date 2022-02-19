# SokobanQ

A [Q-learning](https://en.wikipedia.org/wiki/Q-learning) implementation for an agent to learn how to beat a Sokoban map.

## Maps

Maps can be created using a simple text editor, using just 5 different characters.
* `A` - Agent
* `X` - Wall
* `.` - Passage
* `B` - Box
* `G` - Goal

Here is an example [map](maps/map.txt):
```
XXXXXXXXX
XX..GXG.X
X.J.XX..X
X.J.J..GX
X......MX
XXXXXXXXX
```

## Demo
<video src='demo.mov' height=400px></video>

## Usage

1. Clone this repo.
2. Create a virtual environment.
3. Source the virtual environment.
4. Install requirements.
5. Create a map (optional)
6. Run the code.

```bash
$ git clone [repo]
$ python3 -m venv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
$ python3 SokobanQ.py
```
### Arguments

```
$ python3 SokobanQ.py -h
usage: SokobanQ.py [-h] [-m MAP]

Start a Q-learning agent on a Sokoban map.

optional arguments:
  -h, --help         show this help message and exit
  -m MAP, --map MAP  set the path to the file containing the map. Default: maps/map.txt.
```
