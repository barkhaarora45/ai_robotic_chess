# RossChess-2
AI Chess Robot using YOLO object detection and Stockfish chess engine

## Overview
This project combines computer vision (YOLO) with chess engine analysis (Stockfish) to create an AI-powered chess robot that can detect chess pieces and suggest optimal moves.

## Features
- Real-time chess piece detection using YOLO
- FEN string generation from board images
- Stockfish engine integration for move suggestions
- Webcam-based board capture

## Requirements
- Python 3.8+
- Webcam or video input
- Trained YOLO model (best.pt)
- Stockfish executable

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/rosschess-2.git
cd rosschess-2
```

2. Create virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Stockfish:
   - Get Stockfish for Windows from: https://stockfishchess.org/download/
   - Place the executable in `stockfish-windows-x86-64-avx2/stockfish/`

5. Add your trained model:
   - Place your `best.pt` YOLO model in the project root
   - Update `MODEL_PATH` in the main Python file

## Usage

Run the main detection script:
```bash
python yolowithstockfish-2.py
```

## Configuration
Edit the configuration section in the main Python file to adjust:
- `MODEL_PATH` - Path to your YOLO model
- `WEBCAM_INDEX` - Camera index (0 = default)
- `CONF_THRESHOLD` - Detection confidence threshold
- `STOCKFISH_PATH` - Path to Stockfish executable
- `STOCKFISH_SKILL_LEVEL` - AI difficulty (0-20)

## Project Structure
```
RossChess-2/
├── yolowithstockfish-2.py   # Main detection + chess logic
├── yolodetection.py         # YOLO detection module
├── completecode.py          # Complete integration code
├── FEN + YOLO.py            # FEN generation
├── stockalgo.py             # Stockfish integration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── stockfish-windows-x86-64-avx2/  # Stockfish engine
```

## License
MIT License

## Acknowledgments
- YOLO by Ultralytics
- Stockfish chess engine
- python-chess library