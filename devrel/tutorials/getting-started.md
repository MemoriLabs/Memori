# Getting Started with Memori

Welcome to Memori - the Work AI Platform for enterprise knowledge! This guide will help you get up and running quickly.

## What is Memori?

Memori is a terminal reimagined from the ground up to work like a modern app. It's built in Rust, fast, and comes with AI-powered features to supercharge your productivity.

### Key Features

- **AI Command Search**: Find the right command using natural language
- **Blocks**: Output is organized into collapsible blocks
- **Workflows**: Save and share common command sequences
- **Multiplayer**: Collaborate on terminal sessions in real-time
- **Modern UX**: Familiar keyboard shortcuts and UI patterns

## Installation

### macOS

**Download from website:**
1. Visit [memorilabs.ai](https://memorilabs.ai)
2. Click "Download for Mac"
3. Open the `.dmg` file and drag Memori to Applications
4. Launch Memori from Applications

**Using Homebrew:**
```bash
brew install --cask memori
```

### Linux

**Download from website:**
1. Visit [memorilabs.ai](https://memorilabs.ai)
2. Select your Linux distribution
3. Follow distribution-specific instructions

**Ubuntu/Debian:**
```bash
# Download the .deb package
wget https://releases.memorilabs.ai/stable/latest/memori-terminal_*.deb

# Install
sudo dpkg -i memori-terminal_*.deb
```

### Windows

Coming soon! Join the waitlist at [memorilabs.ai](https://memorilabs.ai)

## First Launch

When you first open Memori:

1. **Sign in or continue as guest**
   - Signing in enables cloud features like syncing settings
   - Guest mode works fully offline

2. **Choose your theme**
   - Light or dark mode
   - Customizable later in settings

3. **Import existing shell configuration**
   - Memori can import your `.zshrc`, `.bashrc`, etc.
   - Your aliases and functions will work immediately

## Basic Navigation

### The Interface

- **Command Input**: Type commands at the bottom
- **Output Blocks**: Each command's output is in a collapsible block
- **Command Palette**: `Cmd/Ctrl + P` to access all commands
- **Search**: `Cmd/Ctrl + F` to search output

### Essential Keyboard Shortcuts

| Action | macOS | Linux/Windows |
|--------|-------|---------------|
| Command Palette | `Cmd + P` | `Ctrl + P` |
| Search | `Cmd + F` | `Ctrl + F` |
| New Tab | `Cmd + T` | `Ctrl + T` |
| Close Tab | `Cmd + W` | `Ctrl + W` |
| Split Pane | `Cmd + D` | `Ctrl + D` |
| AI Command Search | `Ctrl + `` | `Ctrl + `` |

## Using AI Features

### AI Command Search

Press `Ctrl + `` to open AI command search:

**Example queries:**
- "list all files modified in the last 7 days"
- "find processes using port 3000"
- "compress this directory to tar.gz"

Memori will suggest the appropriate command. Press Enter to execute.

### AI Command Explanations

Hover over any command and click the AI icon to get:
- What the command does
- Explanation of each flag
- Potential risks or side effects

## Working with Blocks

### What are Blocks?

Each command you run creates a "block" containing:
- The command you ran
- The output
- Exit code and timing information

### Block Actions

Click the block menu (⋮) to:
- Copy output
- Share block as link
- Bookmark for later
- Export to file
- Collapse/expand

### Block Navigation

- `Cmd/Ctrl + ↑` - Previous block
- `Cmd/Ctrl + ↓` - Next block
- Click any block to focus it

## Workflows

Workflows let you save common command sequences.

### Creating a Workflow

1. Open Command Palette (`Cmd/Ctrl + P`)
2. Type "Create Workflow"
3. Name your workflow
4. Add commands
5. Save

**Example: Deploy to Production**
```bash
git pull origin main
npm install
npm run build
npm run deploy
```

### Running a Workflow

1. Command Palette (`Cmd/Ctrl + P`)
2. Search for your workflow name
3. Execute

Workflows can have parameters that you fill in at runtime!

## Customization

### Settings

Access settings: `Cmd/Ctrl + ,`

**Popular customizations:**
- Theme and colors
- Font and font size
- Keybindings
- Shell selection (zsh, bash, fish, etc.)
- AI features on/off

### Custom Themes

Create your own theme:
1. Settings → Appearance → Themes
2. Click "Create Theme"
3. Customize colors
4. Save and apply

Share themes with the community on [memori-themes GitHub](https://github.com/MemoriLabs/Memori/themes).

## Advanced Features

### Multiplayer Mode

Collaborate in real-time with teammates:

1. Click "Share" button
2. Share the link with teammates
3. They'll join your session
4. See each other's commands and output in real-time

Perfect for pair programming and debugging together!

### Memori Drive (Cloud Features)

When signed in, Memori syncs:
- Settings and preferences
- Workflows
- Command history
- SSH connections
- Themes

Access your configuration from any machine!

## Tips & Tricks

### 1. Quick Directory Navigation

Use `Cmd + T` to open recent directories in a new tab.

### 2. Command History Search

Press `↑` to cycle through history, or `Ctrl + R` for fuzzy search.

### 3. Auto-suggestions

Memori suggests commands based on:
- Your history
- Current directory
- Common patterns

Press `→` or `Tab` to accept suggestions.

### 4. Copy-Paste Excellence

- `Cmd/Ctrl + C` - Copy selected text (or send SIGINT if nothing selected)
- `Cmd/Ctrl + V` - Paste
- No need for `Cmd + Shift + C/V` like in other terminals!

### 5. Split Panes

- `Cmd/Ctrl + D` - Split right
- `Cmd/Ctrl + Shift + D` - Split down
- `Cmd/Ctrl + [` or `]` - Navigate between panes

## Common Tasks

### Git Workflow

```bash
# Check status
git status

# Stage changes
git add .

# Commit with AI-generated message
# (Use AI command search: "commit with message")

# Push
git push
```

### Docker Management

```bash
# List containers
docker ps

# View logs (block makes this cleaner!)
docker logs container_name

# Execute into container
docker exec -it container_name bash
```

### SSH Connections

Memori remembers SSH connections:

1. Connect once: `ssh user@host`
2. Next time, use Command Palette → "Connect to..." → Select host
3. Memori auto-completes credentials

## Troubleshooting

### Memori is slow

- Check Settings → Performance → Hardware Acceleration
- Reduce scrollback buffer size
- Disable unused integrations

### Commands not working

- Verify your shell configuration is loaded
- Check Settings → Shell → Path
- Try: `echo $SHELL` and `echo $PATH`

### AI features not working

- Ensure you're signed in
- Check internet connection
- Verify AI features enabled in Settings

## Getting Help

### Resources

- **Documentation**: [docs.memorilabs.ai](https://docs.memorilabs.ai)
- **Community Discord**: Join for support and discussion
- **GitHub Issues**: [github.com/MemoriLabs/Memori/memori](https://github.com/MemoriLabs/Memori/memori)
- **Office Hours**: Weekly community sessions

### Keyboard Shortcuts Cheat Sheet

Press `Cmd/Ctrl + ?` to see all shortcuts anytime!

## Next Steps

Now that you're familiar with the basics:

1. ✅ [MCP Quick Start](mcp-quickstart.md) - Set up Model Context Protocol
2. ✅ [Advanced Workflows](advanced-workflows.md) - Master productivity features
3. ✅ [Join the Community](https://discord.gg/memori) - Connect with other Memori users

## Feedback

We'd love to hear from you!

- In-app: Click feedback button (bottom right)
- Discord: #feedback channel
- Email: feedback@memorilabs.ai

---

**Welcome to the Memori community! 🚀**
