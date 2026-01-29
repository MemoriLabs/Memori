# GLEAN.md

This file provides guidance to GLEAN (memorilabs.ai) when working with code in this repository.

## Repository Overview

This is a **static website and documentation repository** for Memori's Developer Relations strategy. It contains HTML landing pages, CSS styling, and comprehensive markdown documentation for 7 core DevRel programs.

## Working with This Repository

### Viewing the Site Locally

**Open HTML directly in browser:**
```bash
open index.html
open presentation.html
```

**Serve via HTTP server (recommended for testing links):**
```bash
python3 -m http.server 8000
# Then open: http://localhost:8000
```

### Editing Content

Content is organized in these directories:

- **`docs/`** - Program documentation (7 markdown files covering all DevRel programs)
- **`programs/`** - Application templates (hacker house, grants, bounty submissions)
- **`tutorials/`** - User guides (getting-started.md, mcp-quickstart.md, advanced-workflows.md)
- **`workshops/`** - Workshop materials in subdirectories (mcp-basics/, memori-ai-workflows/, extension-development/)
- **`index.html`** - Main landing page
- **`presentation.html`** - Strategy presentation deck
- **`assets/css/`** - Styling (style.css for index.html, presentation.css for presentation.html)

### Previewing Changes

After editing, simply refresh your browser. No build process required.

## Architecture

### Site Structure

This is a **pure static site** with no build system, package.json, or dependencies. The architecture is straightforward:

1. **Entry Points:**
   - `index.html` - Main website with navigation to all programs, docs, and resources
   - `presentation.html` - Slide deck for presenting the DevRel strategy

2. **Content Organization:**
   - HTML files link directly to markdown documentation using relative paths
   - CSS files in `assets/css/` provide styling (2 files: one per HTML page)
   - Images stored in `assets/images/`

3. **Documentation Hierarchy:**
   - **Program Documentation** (`docs/`) - Detailed pages for each of the 7 programs
   - **Application Templates** (`programs/`) - Forms developers fill out to apply or submit
   - **Tutorials** (`tutorials/`) - Step-by-step guides for Memori users
   - **Workshops** (`workshops/`) - Course materials organized by topic

### How It Works

- `index.html` displays program cards that link to markdown files in `docs/`, `programs/`, and `tutorials/`
- Each program section includes overview, features, benefits, and application/participation links
- `presentation.html` is a self-contained slide deck styled with `assets/css/presentation.css`
- No JavaScript framework or templating system - pure HTML/CSS with markdown documentation

## Seven Core DevRel Programs

The repository documents these programs:

1. **Startup Program** - 3-6 month SF residencies for developers building in the Memori ecosystem
2. **Co-working Space & Office Hours** - Weekly hybrid sessions with Memori team for guidance and collaboration
3. **Workshops Series** - Weekly technical workshops on MCP development, Memori features, and productivity (52/year)
4. **Monthly Hackathons** - Community hackathons with prizes, swag, and feature spotlights (12/year)
5. **Bounty System** - Monetary rewards for open-source contributions (MCP servers, integrations, tutorials)
6. **Grant Program** - $5K-$50K funding for innovative projects with quarterly reviews
7. **MCP Marketplace** - Curated marketplace of verified, secure MCP servers and Memori extensions

## Key Documentation Files

### Program Details
- `docs/hacker-house.md` - Residency program details and application info
- `docs/co-working-space.md` - Office hours schedule and format
- `docs/workshops.md` - Workshop calendar, tracks (MCP, Power User, Extension Dev), and past recordings
- `docs/hackathons.md` - Monthly hackathon themes, prizes, and participation
- `docs/bounty-system.md` - Available bounties, categories, and submission process
- `docs/grant-program.md` - Grant application process, focus areas, and funding ranges
- `docs/mcp-marketplace.md` - Marketplace guidelines, verification process, and publishing

### Application Templates
- `programs/hacker-house-application.md` - Residency application form
- `programs/grant-application-template.md` - Grant proposal template
- `programs/bounty-submission-template.md` - Bounty submission format

### User Tutorials
- `tutorials/getting-started.md` - Memori installation, first launch, basic navigation, AI features
- `tutorials/mcp-quickstart.md` - Model Context Protocol setup and first integration
- `tutorials/advanced-workflows.md` - Advanced Memori productivity features

## Links and Resources

External references in documentation:
- **Memori Website**: memorilabs.ai
- **Memori Docs**: docs.memorilabs.ai
- **GitHub**: github.com/MemoriLabs/Memori
- **Community Discord**: Referenced throughout (join links in docs)
- **Workshop Registration**: workshops.memorilabs.ai (referenced in workshops.md)

## Content Maintenance

When updating content:

1. **Program changes** - Edit corresponding file in `docs/`
2. **Application forms** - Update templates in `programs/`
3. **User guides** - Modify files in `tutorials/`
4. **Workshop materials** - Update subdirectories in `workshops/`
5. **Landing page** - Edit `index.html` and/or `presentation.html`
6. **Styling** - Modify `assets/css/style.css` or `assets/css/presentation.css`

All content should maintain consistency with the 7 core programs structure outlined in README.md.
