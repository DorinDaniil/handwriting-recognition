# fonts

Рукописные кириллические шрифты (`.ttf`/`.otf`). Сами файлы не коммитятся (`.gitignore`).

Наполнить:
```
python scripts/fetch_fonts.py
```
Вручную: handwriting+cyrillic с fontesk.com / localfonts.eu / fontspace.com — положить сюда.

`FontBank` проверяет покрытие глифов (fontTools), отбрасывает шрифты ниже
`min_glyph_coverage`, кэширует результат в `_coverage.json`.
