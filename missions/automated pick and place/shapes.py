import pybullet as p

def create_sphere(radius, mass, base_position, base_orientation): 

    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius = radius)

    sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius = radius)

    sphere_id = p.createMultiBody(
        baseMass = mass, 
        baseCollisionShapeIndex = sphere_collision, 
        baseVisualShapeIndex = sphere_visual, 
        basePosition = base_position, 
        baseOrientation = base_orientation
    )

    return sphere_id