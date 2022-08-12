import re
import numpy as np



def get_vertices_and_edges(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            lines.append(line.strip('\n'))

    contents = ''.join(lines)
    pos_strings = re.findall('msh\.POS[ ]*=[ ]*\[[\-;e\.0-9 ]*\]', contents)
    pos_string_list = pos_strings[0].strip('msh.POS = [').strip(']').split(';')
    vertices_list = []
    for pos in pos_string_list:
        pos_list = pos.split(' ')
        pos2 = [float(pos) for pos in pos_list if pos != '']
        if len(pos2) == 3:
            vertices_list.append(pos2)
    vertices = np.array(vertices_list)

    tet_strings = re.findall('msh\.TETS[ ]*=[ ]*\[[0-9 ]*\]', contents)
    tet_string_list = tet_strings[0].strip('msh.TETS =[').strip(']').split(' ')
    tet_flat_list = [int(val) for val in tet_string_list if val.isnumeric()]
    tet_list = []
    for i in range(0, len(tet_flat_list), 5):
        a, b, c, d = tet_flat_list[i], tet_flat_list[i+1], tet_flat_list[i+2],\
            tet_flat_list[i+3]
        tet_list.append([a, b, c, d])
    tetrahedrons = np.array(tet_list)

    return vertices, tetrahedrons

if __name__ == '__main__':
    vertices, tetrahedrons = get_vertices_and_edges('./data/cylinder.txt')
    N = vertices.shape[0]
    vertices_dist = np.zeros([N, 4])
    for i in range(vertices.shape[0]):
        vertices_dist[i, 0:3] = vertices[i]
        vertices_dist[i, 3] = np.linalg.norm(vertices[i])
    print(vertices_dist[2000:2010])
    print(vertices.shape)
    print(np.amax(tetrahedrons))
    print(np.amin(tetrahedrons))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    ax.scatter(vertices.T[0], vertices.T[1], vertices.T[2], alpha=0.1)
    plt.show()
    plt.close()
