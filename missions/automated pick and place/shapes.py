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

def create_cylinder(radius, height, mass, base_position, base_orientation): 

    cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius = radius, height = height)

    cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius = radius, length = height)

    cylinder_id = p.createMultiBody(
        baseMass = mass, 
        baseCollisionShapeIndex = cylinder_collision, 
        baseVisualShapeIndex = cylinder_visual, 
        basePosition = base_position, 
        baseOrientation = base_orientation
    )

    return cylinder_id

def create_cuboid(width, length,  height, mass, base_position, base_orientation): 

    cuboid_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents = [width / 2, length / 2, height / 2])

    cuboid_visual = p.createVisualShape(p.GEOM_BOX, halfExtents = [width / 2, length / 2, height / 2])

    cuboid_id = p.createMultiBody(
        baseMass = mass, 
        baseCollisionShapeIndex = cuboid_collision, 
        baseVisualShapeIndex = cuboid_visual, 
        basePosition = base_position, 
        baseOrientation = base_orientation
    )

    return cuboid_id