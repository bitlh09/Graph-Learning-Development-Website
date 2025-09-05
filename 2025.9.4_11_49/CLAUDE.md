# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph-Learning-Development-Website is an educational platform for graph neural networks (GNNs) that combines interactive tutorials, hands-on programming environments, and visual learning experiences. The codebase serves as both a frontend educational site and backend for handling graph datasets and training algorithms.

## Key Architecture Components

### Frontend Structure
- **HTML pages**: `index.html`, `tutorials.html`, `playground.html`, `community.html`, `resources.html`, `game.html`, `team.html`
- **CSS**: Tailwind CSS for styling with custom theme in `css/styles.css`
- **JavaScript**: Interactive features in `js/` directory
- **Visualization**: Uses D3.js, Plotly.js for graph visualizations and interactive demos

### Backend & Data Processing
- **Python services**: 
  - `cora_server.py`: Flask/CORS service for Cora dataset processing
  - `resources_server.js`: Express.js server for download serving
  - Graph training implementations: `graphsage_complete.py`, `graphsage_training.py`
- **Graph datasets**: Cora and Citeseer datasets in `data/` directory

### Educational Components
- **Progressive learning**: Life scenarios → formal concepts → algorithmic principles
- **Interactive coding**: Pyodide-based Python environment in browser
- **Visual learning**: Graph structure changes, attention weights, training process
- **Game mechanics**: Quiz systems and achievement-based learning

## Commands

### Web Development
```bash
# Start resources server
npm start

# Start Cora data server
python cora_server.py

# Serve files (if needed)
python -m http.server 8000
```

### Graph Learning Python Scripts
```bash
# Test GraphSAGE implementation
python graphsage_complete.py
python test_graphsage.py
python graphsage_training.py

# Test CORA GCN
python test_numpy_gcn.py
```

### Dataset & Training Scripts
```bash
# Run GraphSAGE with real browser backend
python graphsage_real_complete.py
python graphsage_real_data.py
python graphsage_real_demo.py
```

### Testing
```bash
# Frontend tests (HTML files)
open test-cora-gcn.html
open test_pyodide.html
open graphsage_test.html
open pyodide_test.html
```

## Key Technology Stack

- **Frontend**: HTML5, Tailwind CSS, JavaScript (ES6), D3.js, Plotly.js
- **Interactive Python**: Pyodide (WebAssembly-based Python runtime)
- **Backend**: Node.js/Express, Python/Flask with CORS
- **Datasets**: Cora, Citeseer (citation networks)
- **ML Libraries**: NumPy-based implementations (GCN, GraphSAGE, GAT, etc.)

## File Organization Patterns

- Core algorithms: Root-level Python files (e.g., `graphsage_complete.py`, `1.py`, `2.py`)
- Educational content: `tutorials.html`, `community.html` 
- Interactive features: `playground.html`, `game.html`
- Server-side: `cora_server.py`, `resources_server.js`
- Visual demos: `graphsage_pyodide.html`, `graphsage_real_browser.html`