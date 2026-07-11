# How to add audio files

Put your audio files under this folder structure:

```text
audio/
├── male/
│   ├── original.wav
│   ├── global.wav
│   ├── token.wav
│   └── smooth21.wav
└── female/
    ├── original.wav
    ├── global.wav
    ├── token.wav
    └── smooth21.wav
```

The filenames must match the paths used in `index.html`.

## Suggested commands

From the website folder:

```bash
mkdir -p audio/male audio/female

cp /path/to/male_original.wav  audio/male/original.wav
cp /path/to/male_global.wav    audio/male/global.wav
cp /path/to/male_token.wav     audio/male/token.wav
cp /path/to/male_smooth21.wav  audio/male/smooth21.wav

cp /path/to/female_original.wav  audio/female/original.wav
cp /path/to/female_global.wav    audio/female/global.wav
cp /path/to/female_token.wav     audio/female/token.wav
cp /path/to/female_smooth21.wav  audio/female/smooth21.wav
```

Then commit and push:

```bash
git add index.html styles.css audio/
git commit -m "Add audio demo website"
git push
```

## Local preview

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```
