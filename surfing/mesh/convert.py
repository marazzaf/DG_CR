import meshio

msh = meshio.read("mesh_1.msh")
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "line":
        line_cells = cell.data

for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "line":
        line_data = msh.cell_data_dict["gmsh:physical"][key]
        
triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells})

meshio.write("mesh_1.xdmf", triangle_mesh)
