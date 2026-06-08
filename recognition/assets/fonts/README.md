# Шрифты для синтетики

Сюда складываются кириллические **рукописные** шрифты (`.ttf`/`.otf`/`.ttc`).
Сами файлы шрифтов в git не коммитятся (см. `.gitignore`).

Наполнить автоматически:

```bash
python scripts/fetch_fonts.py
```

Или вручную — скачать бесплатные шрифты со стилем *handwriting + cyrillic*:
[fontesk.com](https://fontesk.com/tag/cyrillic/), [localfonts.eu](https://localfonts.eu/freefonts/handwritten-cyrillic-free-fonts/),
[fontspace.com](https://www.fontspace.com/category/handwriting,cyrillic) — и положить `.ttf` сюда.

`FontBank` сам проверит покрытие глифов (через fontTools), отбросит латиница-only/только-заглавные
шрифты и закэширует результат в `_coverage.json`. Хотите больше разнообразия — просто добавьте
файлы и перезапустите.
