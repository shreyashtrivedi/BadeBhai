# VHS script for the terminal demo: `vhs docs/demo.tape` -> docs/demo.gif
# Needs: docker compose stack running (or local `uv run badebhai serve`); runs fully on the local model.

Output docs/demo.gif
Set FontSize 15
Set Width 1200
Set Height 720
Set Theme "Catppuccin Latte"
Set TypingSpeed 35ms

Type "uv run badebhai repos" Enter
Sleep 3s

Type "uv run badebhai search psf/requests 'where is authentication handled' -k 4" Enter
Sleep 6s

Type "uv run badebhai ask psf/requests 'Why was the retry logic added, and is it related to any open issues?'" Enter
Sleep 45s

Type "uv run badebhai ask psf/requests 'Where is the GraphQL resolver?'" Enter
Sleep 40s

Type "uv run badebhai eval --retrieval-only" Enter
Sleep 25s
