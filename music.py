Urllist = [
    "https://www.youtube.com/watch?v=2RicaUqd9Hg",
    "https://www.youtube.com/watch?v=-CCgDvUM4TM",
    "https://www.youtube.com/watch?v=9KhbM2mqhCQ",
]

output_folder = "data"
for url in urllist:
    !youtube-dl --extract-audio --audio-format wav --audio-quality 0 --output "{output_folder}/%(id)s.%(ext)s" "{url}"