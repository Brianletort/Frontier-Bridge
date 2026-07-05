# VHS tape for the README demo GIF: detect -> plan -> refusal.
# Reproduce with: vhs docs/demo.tape   (from the repo root, venv installed)

Output docs/assets/demo.gif

Set Theme "Catppuccin Mocha"
Set FontSize 15
Set Width 1180
Set Height 580
Set Padding 16
Set TypingSpeed 35ms

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 1s
Show

Type "# What can this machine actually run? Profile it first."
Enter
Sleep 800ms
Type "frontier detect -o hardware_profiles/local/demo.yaml && grep -A5 'kind: storage' hardware_profiles/local/demo.yaml"
Enter
Sleep 10s

Type "clear"
Enter
Type "# Plan a 107 GB MoE model against the measured profile."
Enter
Sleep 800ms
Type "frontier plan deepseek-v4-flash --quant q2_imatrix --engine llama_cpp --hardware apple_m5_max_137gb_detected --workload coding_agent --ctx 32768 | head -28"
Enter
Sleep 6s

Type "clear"
Enter
Type "# And when the numbers don't work, it says no out loud."
Enter
Sleep 800ms
Type "frontier plan glm-5.2 --hardware rtx6000_96gb_64ram --workload long_context --ctx 2000000"
Enter
Sleep 6s
