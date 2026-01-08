# LAMPORT

```
██╗      █████╗ ███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗
██║     ██╔══██╗████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝
██║     ███████║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║
██║     ██╔══██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║
███████╗██║  ██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝
```

**AI-powered Solana smart contract generator using Anchor and Rust**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

Lamport is an AI-powered CLI tool that generates production-ready Solana smart contracts from natural language specifications. It uses a multi-agent LangGraph pipeline to interpret your requirements, plan the project structure, generate Rust/Anchor code, and validate/build the contract automatically.

Simply describe what you want in plain English, and Lamport creates the complete Anchor project with working Rust code.

## Features

- **Natural Language Processing** - Convert plain English specs to structured token specifications
- **Automated Project Scaffolding** - Creates complete Anchor project structure (Anchor.toml, Cargo.toml, lib.rs)
- **Production-Ready Code** - Generates idiomatic Rust code following Anchor 0.30.x best practices
- **Multi-Agent Pipeline** - Specialized agents for interpretation, planning, generation, and validation
- **Built-in Debugger** - Automatically fixes build and validation errors
- **Interactive Mode** - Conversational interface for iterative contract development

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Lamport Pipeline                               │
└─────────────────────────────────────────────────────────────────────────┘

  User Spec ──► spec_interpreter ──► project_planner ──► code_generator
                 (NLP → TokenSpec)     (Anchor Scaffold)   (Rust Code)
                                                                  │
                                                                  ▼
                                    ┌──────────────────────────────┐
                                    │     static_validator         │
                                    │  (Rust syntax, Anchor check) │
                                    └──────────────────────────────┘
                                              │       │
                                              │       ▼
                                              │   ┌─────────┐
                                              │   │ Success │────► Build Contract
                                              │   └─────────┘
                                              │
                                              ▼
                                          ┌─────────┐
                                          │Debugger │────► Fix & Retry
                                          └─────────┘
                                              │
                                              ▼
                                          ┌─────────┐
                                          │  Abort  │
                                          └─────────┘
```

### Agents

| Agent | Purpose |
|-------|---------|
| `spec_interpreter` | Converts natural language → structured `TokenSpec` |
| `project_planner` | Creates Anchor project scaffold |
| `file_planner` | Plans file structure and module organization |
| `code_generator` | Generates Rust instruction handlers |
| `debugger` | Fixes build/validation errors |

## Prerequisites

- **Rust & Cargo** - [Install via rustup](https://rustup.rs/)
- **Anchor Version 0.30+** - [Installation Guide](https://www.anchor-lang.com/docs/installation)
- **Python 3.13+**
- **OpenRouter API Key** - [Get one here](https://openrouter.ai/settings/keys)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-contractor.git
cd smart-contractor

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Install the CLI (optional)
pip install -e .
```

## Configuration

Create a `.env` file:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

Optional configuration in `config.yaml`:

```yaml
models:
  spec_interpreter: "x-ai/grok-code-fast-1"
  project_planner: "x-ai/grok-code-fast-1"
  code_generator: "x-ai/grok-code-fast-1"
  debugger: "x-ai/grok-code-fast-1"

output:
  colors: true
  ascii_art: true
  theme: "dark"
```

## Usage

### CLI Mode

Generate a contract from a natural language description:

```bash
solana-contractor generate "create a mintable token called MyToken"
```

Test mode (uses MockLLM, no API calls):

```bash
solana-contractor generate "create a token" --test
```

### Interactive Mode

Start a conversational session:

```bash
make run
# or
python src/main.py
```

### System Check

Verify prerequisites (Rust, Cargo, Anchor):

```bash
solana-contractor check
```

## Examples

### Basic Token Contract

```bash
solana-contractor generate "create a mintable token called MyToken with symbol MYT and 9 decimals"
```

### Custom Program

```bash
solana-contractor generate "create a staking program where users can stake tokens and earn rewards"
```

### Swap Contract

```bash
solana-contractor generate "create a simple token swap contract with two token accounts"
```

## Project Structure

Generated projects follow Anchor conventions:

```
contracts/<name>_<timestamp>/
├── Anchor.toml           # Anchor configuration
├── programs/
│   └── <name>/
│       ├── Cargo.toml    # Rust dependencies
│       └── src/
│           └── lib.rs    # Contract instructions
├── tests/
│   └── <name>.ts         # TypeScript tests
└── migrations/
    └── deploy.ts         # Deployment script
```

## Development

```bash
# Install dependencies
make install

# Linting and formatting
make lint     # Check with ruff
make fix      # Auto-fix issues

# Run tests
uv run pytest

# Run CLI
solana-contractor generate "your spec here"
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with Anchor, Rust, and LangGraph**
