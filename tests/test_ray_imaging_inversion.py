import numpy as np
from geowave.ray.layered import Layer,reflected_ray
from geowave.ray.bending import GridVelocity,bend_path,path_time
from geowave.ray.travel_table import fast_grid_travel_times
from geowave.imaging.kirchhoff import constant_velocity_migration
from geowave.inversion.impedance import reflectivity_from_impedance,recursive_impedance
from geowave.inversion.avo import fit_avo
from geowave.inversion.tomography import TravelTimeObservation,straight_ray_tomography

def test_layered_ray():
    r=reflected_ray([Layer(500,1800),Layer(500,2500)],1000)
    assert r.time_s>0 and r.offset_m>900

def test_bending_and_table():
    x=np.linspace(0,1000,31);z=np.linspace(0,1000,31);v=np.full((31,31),2000.0);v[15:]+=500
    grid=GridVelocity(x,z,v);path=bend_path(grid,(0,0),(1000,1000),nodes=15,iterations=5)
    assert path_time(grid,path)>0
    table=fast_grid_travel_times(grid,0,0)
    assert table.time_s[-1,-1]>0

def test_kirchhoff(dataset):
    img=constant_velocity_migration(dataset,np.linspace(0,150,8),np.linspace(50,700,16),2000,aperture_m=300)
    assert img.amplitude.shape==(16,8)
    assert np.isfinite(img.amplitude).all()

def test_impedance_roundtrip():
    z=np.array([4000,4500,5000,4300.])
    r=reflectivity_from_impedance(z);back=recursive_impedance(r,z[0])
    assert np.allclose(z,back)

def test_avo():
    angles=np.array([0,10,20,30,40.]);amp=.2-.15*np.sin(np.deg2rad(angles))**2
    fit=fit_avo(angles,amp)
    assert fit.r2>.99

def test_tomography():
    obs=[TravelTimeObservation(0,100,300,.32),TravelTimeObservation(100,300,300,.34),TravelTimeObservation(0,300,400,.5)]
    r=straight_ray_tomography(obs,np.array([0,100,200,300]),2000,damping=.1)
    assert len(r.slowness)==3
