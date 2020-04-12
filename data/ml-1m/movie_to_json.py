import json

with open("movies.dat", "r") as f:
    lines = f.readlines()
    movies_list = []
    for line in lines:
        _, movie, _ = line.split("::")
        movies_list.append(movie)
    with open("movies.json", "w") as j:
        j.write(json.dumps(movies_list))
