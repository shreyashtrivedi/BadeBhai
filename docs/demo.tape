# VHS script for the terminal demo: `vhs docs/demo.tape` -> docs/demo.gif
# Needs: docker compose stack running (or local `uv run repomind serve`), ANTHROPIC_API_KEY in .env.

Output docs/demo.gif
Set FontSize 15
Set Width 1200
Set Height 720
Set Theme "Catppuccin Latte"
Set TypingSpeed 35ms

Type "uv run repomind repos" Enter
Sleep 3s

Type "uv run repomind search psf/requests 'where is authentication handled' -k 4" Enter
Sleep 6s

Type "uv run repomind ask psf/requests 'Why was the retry logic added, and is it related to any open issues?'" Enter
Sleep 45s

Type "uv run repomind ask psf/requests 'Where is the GraphQL resolver?'" Enter
Sleep 40s

Type "uv run repomind eval --retrieval-only" Enter
Sleep 25s
