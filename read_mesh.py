import re


def read_mesh(fname):
    vertices = {}
    faces = {}
    with open(fname, 'r') as f:
        is_vertices = False
        i = 0
        j = 0
        for line in f:
            numbers = re.findall('[0-9]+\.*[0-9]*', line)
            if len(numbers) == 3: # get vertex rows
                i += 1
                numbers = [float(k) for k in numbers]
                vertices[i] = numbers
            if len(numbers) == 4: # get face rows
                j += 1
                numbers = [int(k) for k in numbers]
                faces[j] = numbers
    return vertices, faces
