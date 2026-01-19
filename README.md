# LUMINO MCP Server

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.10%2B-green.svg)](https://modelcontextprotocol.io/)

An open source MCP (Model Context Protocol) server providing AI-powered tools for Kubernetes, OpenShift, and Tekton monitoring, analysis, and troubleshooting.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Available Tools](#available-tools)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [MCP Client Integration](#mcp-client-integration)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

LUMINO MCP Server transforms how Site Reliability Engineers (SREs) and DevOps teams interact with Kubernetes clusters. By exposing 37 specialized tools through the Model Context Protocol, it enables AI assistants to:

- **Monitor** cluster health, resources, and pipeline status in real-time
- **Analyze** logs, events, and anomalies using statistical and ML techniques
- **Troubleshoot** failed pipelines with automated root cause analysis
- **Predict** resource bottlenecks and potential issues before they occur
- **Simulate** configuration changes to assess impact before deployment

## Features

### Kubernetes & OpenShift Operations
- Namespace and pod management
- Resource querying with flexible output formats
- Label-based resource search across clusters
- OpenShift operator and MachineConfigPool status
- etcd log analysis

### Tekton Pipeline Intelligence
- Pipeline and task run monitoring across namespaces
- Detailed log retrieval with optional cleaning
- Failed pipeline root cause analysis
- Cross-cluster pipeline tracing
- CI/CD performance baselining

### Advanced Log Analysis
- Smart log summarization with configurable detail levels
- Streaming analysis for large log volumes
- Hybrid analysis combining multiple strategies
- Semantic search using NLP techniques
- Anomaly detection with severity classification

### Predictive & Proactive Monitoring
- Statistical anomaly detection using z-score analysis
- Predictive log analysis for early warning
- Resource bottleneck forecasting
- Certificate health monitoring with expiry alerts
- TLS certificate issue investigation

### Event Intelligence
- Smart event retrieval with multiple strategies
- Progressive event analysis (overview to deep-dive)
- Advanced analytics with ML pattern detection
- Log-event correlation

### Simulation & What-If Analysis
- Monte Carlo simulation for configuration changes
- Impact analysis before deployment
- Risk assessment with configurable tolerance
- Affected component identification

## Quick Start

Get started with LUMINO in under 2 minutes:

### For Claude Code CLI Users (Easiest)

Simply ask Claude Code:

```
Provision the Lumino MCP server from https://github.com/spre-sre/lumino-mcp-server as an MCP integration for this Claude Code project
```

Claude will automatically clone, configure, and integrate the server into your project.

### For Other MCP Clients

Choose your preferred installation method:
- **MCPM (Recommended)**: `mcpm install @spre-sre/lumino-mcp-server`
- **Manual Setup**: See detailed [MCP Client Integration](#mcp-client-integration) instructions

### Verify Installation

Once installed, test with a simple query:

```
"List all namespaces in my Kubernetes cluster"
```

## Prerequisites

### Required
- **Python 3.10 or higher** - Core runtime
- **MCP Client** - One of:
  - [Claude Desktop](https://claude.ai/download)
  - [Claude Code CLI](https://github.com/anthropics/claude-code)
  - [Gemini CLI](https://github.com/google/generative-ai-cli)
  - [Cursor IDE](https://cursor.sh/)

### For Kubernetes Features
- **Kubernetes/OpenShift Access** - Valid kubeconfig with read permissions
- **RBAC Permissions** - Ability to list pods, namespaces, and other resources

### Optional (Recommended)
- **[uv](https://docs.astral.sh/uv/)** - Faster dependency management than pip
- **[MCPM](https://github.com/spre-sre/mcpm)** - Easiest installation experience
- **Prometheus** - For advanced metrics and forecasting features

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/spre-sre/lumino-mcp-server.git
cd lumino-mcp-server

# Install dependencies
uv sync

# Run the server
uv run python main.py
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/spre-sre/lumino-mcp-server.git
cd lumino-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the server
python main.py
```

## Usage

### Local Mode (stdio transport)

By default, the server runs in local mode using stdio transport, suitable for direct integration with MCP clients:

```bash
python main.py
```

### Kubernetes Mode (HTTP streaming transport)

When running inside Kubernetes, set the namespace environment variable to enable HTTP streaming:

```bash
export KUBERNETES_NAMESPACE=my-namespace
python main.py
```

The server automatically detects the environment and switches transport modes.

## Usage Examples

### 🔍 Intelligent Root Cause Analysis

Investigate and diagnose complex failures with automated analysis:

```
"Generate a comprehensive RCA report for the failed pipeline run 'build-api-pr-456' in namespace ci-cd"
```

```
"Analyze what caused pod crashes in namespace production over the last 6 hours and correlate with resource events"
```

```
"Investigate the TLS certificate issues affecting services in namespace ingress-nginx"
```

### 🎯 Predictive Intelligence & Forecasting

Anticipate problems before they impact your systems:

```
"Predict resource bottlenecks across all production namespaces for the next 48 hours"
```

```
"Analyze historical pipeline performance and detect anomalies in build times for the last 30 days"
```

```
"Check cluster certificate health and alert me about any certificates expiring in the next 60 days"
```

```
"Use predictive log analysis to identify potential failures in namespace monitoring before they occur"
```

### 🧪 Simulation & What-If Analysis

Test changes safely before applying them to production:

```
"Simulate the impact of increasing memory limits to 4Gi for all pods in namespace backend-services"
```

```
"Run a what-if scenario for scaling deployments to 10 replicas and analyze resource consumption"
```

```
"Simulate configuration changes for nginx ingress controller and assess risk to existing traffic"
```

### 🗺️ Topology & Dependency Mapping

Understand system architecture and component relationships:

```
"Generate a live topology map of all services, deployments, and their dependencies in namespace microservices"
```

```
"Map the complete dependency graph for the payment-service including all connected resources"
```

```
"Show me the topology of components affected by the cert-manager service"
```

### 🔬 Advanced Investigation & Forensics

Deep-dive into complex issues with multi-faceted analysis:

```
"Perform an adaptive namespace investigation for production - analyze logs, events, and resource patterns"
```

```
"Create a detailed investigation report for resource constraints and bottlenecks in namespace data-processing"
```

```
"Trace pipeline execution for commit SHA abc123def from source to deployment across all namespaces"
```

```
"Search logs semantically for 'authentication failures related to expired tokens' across the last 24 hours"
```

### 📊 CI/CD Pipeline Intelligence

Optimize and troubleshoot your continuous delivery pipelines:

```
"Establish performance baselines for all Tekton pipelines and flag runs deviating by more than 2 standard deviations"
```

```
"Trace the complete pipeline flow for image 'api:v2.5.3' from build to production deployment"
```

```
"Analyze failed pipeline runs in namespace tekton-pipelines and identify common failure patterns"
```

```
"Compare current pipeline run times against 30-day baseline and highlight performance degradation"
```

### 🎨 Progressive Event Analysis

Multi-level event investigation from overview to deep-dive:

```
"Start with an overview of events in namespace kube-system, then drill down into critical issues"
```

```
"Perform advanced event analytics with ML pattern detection for namespace monitoring over the last 12 hours"
```

```
"Correlate events with pod logs to identify the root cause of CrashLoopBackOff in namespace applications"
```

### 🚀 Real-Time Monitoring & Alerts

Stay informed about cluster health and pipeline status:

```
"Show me the status of all Tekton pipeline runs cluster-wide and highlight long-running pipelines"
```

```
"List all failed TaskRuns in the last hour with error details and recommended actions"
```

```
"Monitor OpenShift cluster operators and alert on any degraded components"
```

```
"Check MachineConfigPool status and show which nodes are being updated"
```

### 🔐 Security & Compliance

Ensure cluster security and certificate management:

```
"Scan all namespaces for expiring certificates and generate a renewal schedule"
```

```
"Investigate TLS certificate issues causing handshake failures in namespace istio-system"
```

```
"Audit all secrets and configmaps for sensitive data exposure patterns"
```

### 📈 Advanced Analytics & ML Insights

Leverage machine learning for pattern detection:

```
"Use streaming log analysis to process large log volumes from namespace data-pipeline with error pattern detection"
```

```
"Detect anomalies in log patterns using ML analysis with medium severity threshold for namespace api-gateway"
```

```
"Analyze resource utilization trends using Prometheus metrics and forecast capacity needs"
```

## Configuration

### Kubernetes Authentication

The server automatically detects Kubernetes configuration:

1. **In-cluster config** - When running inside a Kubernetes pod
2. **Local kubeconfig** - When running locally (uses `~/.kube/config`)

### Environment Variables

| Variable | Description | Default | When to Use |
|----------|-------------|---------|-------------|
| `KUBERNETES_NAMESPACE` | Namespace for K8s mode | - | When running server inside a Kubernetes pod |
| `K8S_NAMESPACE` | Alternative namespace variable | - | Alternative to `KUBERNETES_NAMESPACE` |
| `PROMETHEUS_URL` | Prometheus server URL for metrics | Auto-detected | Custom Prometheus endpoint or non-standard port |
| `KUBECONFIG` | Path to kubeconfig file | `~/.kube/config` | Multiple clusters or custom kubeconfig location |
| `LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `INFO` | Debugging issues or reducing log noise |
| `MCP_SERVER_LOG_LEVEL` | MCP framework log level | `INFO` | Troubleshooting MCP protocol issues |
| `PYTHONUNBUFFERED` | Disable Python output buffering | - | Recommended for MCP clients to see real-time logs |

## Available Tools

### Kubernetes Core (4 tools)

| Tool | Description |
|------|-------------|
| `list_namespaces` | List all namespaces in the cluster |
| `list_pods_in_namespace` | List pods with status and placement info |
| `get_kubernetes_resource` | Get any Kubernetes resource with flexible output |
| `search_resources_by_labels` | Search resources across namespaces by labels |

### Tekton Pipelines (6 tools)

| Tool | Description |
|------|-------------|
| `list_pipelineruns` | List PipelineRuns with status and timing |
| `list_taskruns` | List TaskRuns, optionally filtered by pipeline |
| `get_pipelinerun_logs` | Retrieve pipeline logs with optional cleaning |
| `list_recent_pipeline_runs` | Recent pipelines across all namespaces |
| `find_pipeline` | Find pipelines by pattern matching |
| `get_tekton_pipeline_runs_status` | Cluster-wide pipeline status summary |

### Log Analysis (6 tools)

| Tool | Description |
|------|-------------|
| `analyze_logs` | Extract error patterns from log text |
| `smart_summarize_pod_logs` | Intelligent log summarization |
| `stream_analyze_pod_logs` | Streaming analysis for large logs |
| `analyze_pod_logs_hybrid` | Combined analysis strategies |
| `detect_log_anomalies` | Anomaly detection with severity levels |
| `semantic_log_search` | NLP-based semantic log search |

### Event Analysis (3 tools)

| Tool | Description |
|------|-------------|
| `smart_get_namespace_events` | Smart event retrieval with strategies |
| `progressive_event_analysis` | Multi-level event analysis |
| `advanced_event_analytics` | ML-powered event pattern detection |

### Failure Analysis & RCA (2 tools)

| Tool | Description |
|------|-------------|
| `analyze_failed_pipeline` | Root cause analysis for failed pipelines |
| `automated_triage_rca_report_generator` | Automated incident reports |

### Resource Monitoring (4 tools)

| Tool | Description |
|------|-------------|
| `check_resource_constraints` | Detect resource issues in namespace |
| `detect_anomalies` | Statistical anomaly detection |
| `prometheus_query` | Execute PromQL queries |
| `resource_bottleneck_forecaster` | Predict resource exhaustion |

### Namespace Investigation (2 tools)

| Tool | Description |
|------|-------------|
| `conservative_namespace_overview` | Focused namespace health check |
| `adaptive_namespace_investigation` | Dynamic investigation based on query |

### Certificate & Security (2 tools)

| Tool | Description |
|------|-------------|
| `investigate_tls_certificate_issues` | Find TLS-related problems |
| `check_cluster_certificate_health` | Certificate expiry monitoring |

### OpenShift Specific (3 tools)

| Tool | Description |
|------|-------------|
| `get_machine_config_pool_status` | MachineConfigPool status and updates |
| `get_openshift_cluster_operator_status` | Cluster operator health |
| `get_etcd_logs` | etcd log retrieval and analysis |

### CI/CD Performance (2 tools)

| Tool | Description |
|------|-------------|
| `ci_cd_performance_baselining_tool` | Pipeline performance baselines |
| `pipeline_tracer` | Trace pipelines by commit, PR, or image |

### Topology & Prediction (2 tools)

| Tool | Description |
|------|-------------|
| `live_system_topology_mapper` | Real-time system topology mapping |
| `predictive_log_analyzer` | Predict issues from log patterns |

### Simulation (1 tool)

| Tool | Description |
|------|-------------|
| `what_if_scenario_simulator` | Simulate configuration changes |

## Architecture

```
lumino-mcp-server/
├── main.py                 # Entry point with transport detection
├── src/
│   ├── server-mcp.py       # MCP server with all 37 tools
│   └── helpers/
│       ├── constants.py    # Shared constants
│       ├── event_analysis.py    # Event processing logic
│       ├── failure_analysis.py  # RCA algorithms
│       ├── log_analysis.py      # Log processing
│       ├── resource_topology.py # Topology mapping
│       ├── semantic_search.py   # NLP search
│       └── utils.py             # Utility functions
└── pyproject.toml          # Project configuration
```

## How It Works

LUMINO acts as a bridge between AI assistants and your Kubernetes infrastructure through the Model Context Protocol:

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Assistant Layer                        │
│          (Claude Desktop, Claude Code CLI, Gemini CLI)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Natural Language Queries
                             │ "Analyze failed pipelines"
                             │ "Predict resource bottlenecks"
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Context Protocol                       │
│                      (MCP Communication)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Tool Invocations & Results
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LUMINO MCP Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Log Analysis │  │ Event Intel  │  │  Predictive  │         │
│  │   (6 tools)  │  │  (3 tools)   │  │  (2 tools)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Pipeline   │  │  Simulation  │  │   Topology   │         │
│  │  (6 tools)   │  │  (1 tool)    │  │  (2 tools)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Kubernetes API Calls
                             │ Prometheus Queries
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes/OpenShift Cluster                  │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Pods   │  │ Services │  │ Tekton   │  │etcd/Logs │       │
│  └──────────┘  └──────────┘  │Pipelines │  └──────────┘       │
│  ┌──────────┐  ┌──────────┐  └──────────┘  ┌──────────┐       │
│  │  Events  │  │ Configs  │  ┌──────────┐  │Prometheus│       │
│  └──────────┘  └──────────┘  │OpenShift │  └──────────┘       │
│                               │Operators │                       │
│                               └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Workflow

1. **User Query** → AI assistant receives natural language request
2. **MCP Translation** → Assistant converts query to appropriate tool calls
3. **LUMINO Processing** → Server executes Kubernetes/Prometheus operations
4. **Data Analysis** → ML/statistical algorithms process raw data
5. **AI Synthesis** → Assistant formats results into human-readable insights

### Key Features

- **Stateless Design** - No data persistence, queries cluster in real-time
- **Automatic Transport Detection** - Switches between stdio (local) and HTTP (K8s) modes
- **Token Budget Management** - Adaptive strategies to handle large log volumes
- **Intelligent Caching** - Smart caching for frequently accessed data
- **Security First** - Uses existing kubeconfig RBAC permissions, no separate auth

## MCP Client Integration

### Method 1: Using MCPM (Recommended for Claude Code CLI / Gemini CLI)

The easiest way to install LUMINO MCP Server for Claude Code CLI or Gemini CLI is using [MCPM](https://github.com/spre-sre/mcpm) - an MCP server package manager.

#### Install MCPM

```bash
# Clone and build MCPM
git clone https://github.com/spre-sre/mcpm.git
cd mcpm
go build -o mcpm .

# Optional: Add to PATH
sudo mv mcpm /usr/local/bin/
```

**Requirements**: Go 1.23+, Git, Python 3.10+, uv (or pip)

#### Install LUMINO MCP Server

```bash
# Install from GitHub repository (short syntax)
mcpm install @spre-sre/lumino-mcp-server

# Or use full GitHub URL
mcpm install https://github.com/spre-sre/lumino-mcp-server.git

# For GitLab repositories (if hosted on GitLab)
mcpm install gl:@spre-sre/lumino-mcp-server

# Install for specific client
mcpm install @spre-sre/lumino-mcp-server --claude  # For Claude Code CLI
mcpm install @spre-sre/lumino-mcp-server --gemini  # For Gemini CLI

# Install globally (works with both Claude and Gemini)
mcpm install @spre-sre/lumino-mcp-server --global
```

**Short syntax explained**:
- `@owner/repo` - Installs from GitHub (default: `https://github.com/owner/repo.git`)
- `gl:@owner/repo` - Installs from GitLab (`https://gitlab.com/owner/repo.git`)
- Full URL - Works with any Git repository

This will:
- Clone the repository to `~/.mcp/servers/lumino-mcp-server/`
- Auto-detect Python project and install dependencies using `uv` (or pip)
- Register with Claude Code CLI or Gemini CLI configuration automatically

#### Manage LUMINO

```bash
# List installed servers
mcpm list

# Update LUMINO
mcpm update lumino-mcp-server

# Remove LUMINO
mcpm remove lumino-mcp-server
```

---

### Method 2: Manual Configuration

If you prefer manual setup or need to configure Claude Desktop / Cursor, follow these client-specific guides:

#### Claude Desktop

1. **Find your config file location**:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Add LUMINO configuration**:

```json
{
  "mcpServers": {
    "lumino": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/lumino-mcp-server",
        "python",
        "main.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Verify**: Look for the hammer icon (🔨) in Claude Desktop to see available tools

---

#### Claude Code CLI

**Option A: Using MCPM** (see Method 1 above)

**Option B: Automatic Provisioning via Claude Code** (Recommended and easiest way)

Simply ask Claude Code to set up the server for you:

```
Provision the Lumino MCP server from https://github.com/spre-sre/lumino-mcp-server as an MCP integration for this Claude Code project
```

Claude will automatically clone, configure, and integrate the server into your project.

**Option C: Manual Configuration**

1. **Find your config file location**:
   - macOS/Linux: `~/.config/claude/mcp_servers.json`
   - Windows: `%APPDATA%\claude\mcp_servers.json`

2. **Add LUMINO configuration**:

```json
{
  "mcpServers": {
    "lumino": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/lumino-mcp-server",
        "python",
        "main.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

3. **Verify installation**:

```bash
# Check MCP servers
claude mcp list

# Test with a query
claude "List all namespaces in my cluster"
```

---

#### Gemini CLI

**Option A: Using MCPM** (Recommended - see Method 1 above)

**Option B: Manual Configuration**

1. **Find your config file location**:
   - macOS/Linux: `~/.config/gemini/mcp_servers.json`
   - Windows: `%APPDATA%\gemini\mcp_servers.json`

2. **Add LUMINO configuration**:

```json
{
  "mcpServers": {
    "lumino": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/lumino-mcp-server",
        "python",
        "main.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

3. **Verify installation**:

```bash
# Check MCP servers
gemini mcp list

# Test with a query
gemini "Show me failed pipeline runs"
```

---

#### Cursor IDE

1. **Open Cursor Settings**:
   - Press `Cmd+,` (macOS) or `Ctrl+,` (Windows/Linux)
   - Search for "MCP" or "Model Context Protocol"

2. **Add MCP Server Configuration**:

In Cursor's MCP settings, add:

```json
{
  "mcpServers": {
    "lumino": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/lumino-mcp-server",
        "python",
        "main.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Alternative - Using Cursor's settings.json**:

1. Open Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
2. Type "Preferences: Open User Settings (JSON)"
3. Add the MCP configuration:

```json
{
  "mcp.servers": {
    "lumino": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/lumino-mcp-server",
        "python",
        "main.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

3. **Restart Cursor IDE**

4. **Verify**: Open Cursor's AI chat and check if LUMINO tools are available

---

### Configuration Notes

**Replace `/path/to/lumino-mcp-server`** with the actual path where you cloned the repository:

```bash
# Example paths:
# macOS/Linux: /Users/username/projects/lumino-mcp-server
# Windows: C:\Users\username\projects\lumino-mcp-server

# If installed via MCPM:
# ~/.mcp/servers/lumino-mcp-server/
```

**Environment Variables** (optional):

Add these to the `env` section if needed:

```json
{
  "env": {
    "PYTHONUNBUFFERED": "1",
    "KUBERNETES_NAMESPACE": "default",
    "PROMETHEUS_URL": "http://prometheus:9090",
    "LOG_LEVEL": "INFO"
  }
}
```

---

### Using Alternative Python Package Managers

#### With pip instead of uv

```json
{
  "command": "python",
  "args": [
    "/path/to/lumino-mcp-server/main.py"
  ]
}
```

**Note**: Ensure you've activated the virtual environment first:

```bash
cd /path/to/lumino-mcp-server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

#### With poetry

```json
{
  "command": "poetry",
  "args": [
    "run",
    "python",
    "main.py"
  ],
  "cwd": "/path/to/lumino-mcp-server"
}
```

---

### Testing Your Configuration

After configuring any client, test the connection:

1. **Check if tools are loaded**:
   - Claude Desktop: Look for 🔨 hammer icon
   - Claude Code CLI: `claude mcp list`
   - Gemini CLI: `gemini mcp list`
   - Cursor: Check AI chat for available tools

2. **Test a simple query**:

```
"List all namespaces in my Kubernetes cluster"
```

3. **Check server logs** (if issues):

```bash
# Run server manually to see errors
cd /path/to/lumino-mcp-server
uv run python main.py
```

Expected output:
```
MCP Server running in stdio mode
Available tools: 37
Waiting for requests...
```

---

### Advanced Configuration

#### Multiple Clusters

Configure multiple LUMINO instances for different clusters:

```json
{
  "mcpServers": {
    "lumino-prod": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/lumino-mcp-server", "python", "main.py"],
      "env": {
        "KUBECONFIG": "/path/to/prod-kubeconfig.yaml"
      }
    },
    "lumino-dev": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/lumino-mcp-server", "python", "main.py"],
      "env": {
        "KUBECONFIG": "/path/to/dev-kubeconfig.yaml"
      }
    }
  }
}
```

#### Custom Log Level

```json
{
  "env": {
    "LOG_LEVEL": "DEBUG",
    "MCP_SERVER_LOG_LEVEL": "DEBUG"
  }
}
```

---

### Supported Transports

The server automatically detects the appropriate transport:

- **stdio** - For local desktop integrations (Claude Desktop, Claude Code CLI, Gemini CLI, Cursor)
- **streamable-http** - For Kubernetes deployments (when `KUBERNETES_NAMESPACE` is set)

## Performance Considerations

### Optimizing for Large Clusters

LUMINO is designed to handle clusters of any size efficiently:

| Cluster Size | Recommendation | Tool Strategy |
|--------------|----------------|---------------|
| **Small** (< 50 pods) | Use default settings | All tools work optimally |
| **Medium** (50-500 pods) | Use namespace filtering | Leverage adaptive tools with auto-sampling |
| **Large** (500+ pods) | Specify time windows and namespaces | Use conservative and streaming tools |
| **Very Large** (1000+ pods) | Combine filters and pagination | Progressive analysis with targeted queries |

### Token Budget Management

LUMINO automatically manages AI context limits:

- **Adaptive Sampling** - Smart tools auto-sample data when volumes are high
- **Progressive Loading** - Stream analysis processes data in chunks
- **Token Budgets** - Configurable limits prevent context overflow
- **Hybrid Strategies** - Automatically selects best analysis approach

### Query Optimization Tips

**Use Namespace Filtering**
```
✅ "Analyze logs for pods in namespace production"
❌ "Analyze all pod logs in the cluster"
```

**Specify Time Windows**
```
✅ "Show events from the last 2 hours"
❌ "Show all events" (might return thousands)
```

**Leverage Smart Tools**
```
✅ "smart_summarize_pod_logs" - Adaptive analysis
❌ Direct log dumps - No processing
```

**Use Progressive Analysis**
```
✅ Start with "overview" → drill down to "detailed"
❌ Jump directly to "deep_dive" on large datasets
```

### Performance Metrics

| Operation | Typical Response Time | Scalability |
|-----------|----------------------|-------------|
| List namespaces | < 1s | O(1) |
| Get pod logs (1 pod) | 1-3s | O(log size) |
| Analyze pipeline run | 2-5s | O(task count) |
| Cluster-wide search | 5-15s | O(namespace count) |
| ML anomaly detection | 3-10s | O(data points) |
| Topology mapping | 5-20s | O(resource count) |

### Caching Strategy

LUMINO uses intelligent caching for frequently accessed data:

- **15-minute cache** - For web-fetched content
- **Session cache** - For hybrid log analysis
- **No persistence** - All data queries cluster in real-time

### Concurrent Requests

The server handles multiple concurrent requests efficiently:

- **Thread-safe operations** - Safe parallel tool execution
- **Connection pooling** - Reuses Kubernetes API connections
- **Async HTTP** - Non-blocking Prometheus queries

### Resource Usage

**Server Resource Requirements**

| Deployment | CPU | Memory | Disk |
|------------|-----|--------|------|
| Local (stdio) | 100-500m | 256-512Mi | Minimal |
| Kubernetes | 200m-1 | 512Mi-1Gi | Minimal |
| High-load | 1-2 | 1-2Gi | Minimal |

**Note**: LUMINO is stateless and requires minimal resources. Most processing happens in the AI assistant.

## Troubleshooting

### Common Issues

**No Kubernetes cluster found**
```
Error: Unable to load kubeconfig
```
Ensure you have a valid kubeconfig at `~/.kube/config` or are running inside a cluster.

**Permission denied for resources**
```
Error: Forbidden - User cannot list resource
```
Check your RBAC permissions. The server needs read access to the resources you want to query.

**Tool timeout**
For large clusters, some tools may timeout. Use filtering options (namespace, labels) to reduce scope.

## Dependencies

- `mcp[cli]>=1.10.1` - Model Context Protocol SDK
- `kubernetes>=32.0.1` - Kubernetes Python client
- `pandas>=2.0.0` - Data analysis
- `scikit-learn>=1.6.1` - ML algorithms
- `prometheus-client>=0.22.0` - Prometheus integration
- `aiohttp>=3.12.2` - Async HTTP client

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting pull requests.

## Security

For security vulnerabilities, please see our [Security Policy](SECURITY.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp) framework
- Inspired by the needs of SRE teams managing complex Kubernetes environments
