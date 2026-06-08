# capstone

Step 1: Muon Generation
- Angular distibution ($\theta$ from zenith, $\phi$ azimuth)
- momenutm spectrum

Step 2: Ray Geometry
- Convert into a 3D direction vector
- Pick a random entry point on the top of the detector
- parameterize the m,uon as a line in 3D space


Step 3: Scattering physics
- define volume as voxels (volume pixels) with materials 
- compute $\theta$ from the formula for each material layer the muon passes
- draw a random angular deflection from a gaussian
- accumulate deflections along the track


Step 4: Detector simulation
- Detector planes (4; 2 about, 2 below sample)
- record hit location of muon for each plane
- add position measurement noise 


Step 5: Track reconstruction
- fit a straight line to the 2 upper hits (incident track)
- fit a straight line to the 2 lower hits (scattered track)
- find the point of closesed approach between the two lines (reconstructed scatter point)

Step 6: Image reconstruction
- create a 3D voxel grid
- for each muon, assign its scattering signal to voxels along its track (with theta^1.5 scaling??? idk why)
- accumulate across all muons

Step 7: visualization
- slice through the 3d voxel grid horizontally 
- display as heatmaps with high-Z objects appearing as bright spots

Step 8: detection decision
- compare signal in candidate voxels againts background
- implement a simple significance threshold