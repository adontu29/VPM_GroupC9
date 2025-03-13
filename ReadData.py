import vtk

# Create a VTK XML PolyData reader
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("dataset/Vortex_Ring_DNS_Re7500_0075.vtp")
reader.Update()  # Read the file

# Get the output data
polydata = reader.GetOutput()

# Print basic information
print(f"Number of points: {polydata.GetNumberOfPoints()}")
print(f"Number of cells: {polydata.GetNumberOfCells()}")

# Iterate through points
for i in range(polydata.GetNumberOfPoints()):
    point = polydata.GetPoint(i)
    print(f"Point {i}: {point}")
