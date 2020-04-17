from sys import argv
from PIL import Image
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 0.0001
REFLECTION_DEPTH = 3


def normalize(vector):
    return vector / np.linalg.norm(vector)


class Ray:
    def __init__(self, start, direction):
        self.start = start
        self.direction = direction


class Surface(ABC):
    @abstractmethod
    def get_surface_normal(self, point=None):
        pass

    @abstractmethod
    def intersect(self, ray, light=False):
        pass


class Plane(Surface):
    def __init__(self, point, normal, color, diffuse_coefficient, reflection_coefficient):
        self.point = np.array(point)
        self.normal = normalize(np.array(normal))
        self.color = np.array(color)
        self.diffuse_coefficient = diffuse_coefficient
        self.reflection_coefficient = reflection_coefficient

    def get_surface_normal(self, point=None):
        return self.normal

    def intersect(self, ray, light=False):
        n_dot_l = np.dot(ray.direction, self.normal)
        if abs(n_dot_l) < 0.001:
            return None
        t = np.dot((self.point - ray.start), self.normal) / n_dot_l
        if light:
            return t if THRESHOLD < t < 1 else None
        return t if t >= THRESHOLD else None


class Sphere(Surface):
    def __init__(self, center, radius, color, diffuse_coefficient, reflection_coefficient):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.diffuse_coefficient = diffuse_coefficient
        self.reflection_coefficient = reflection_coefficient

    def get_surface_normal(self, point=None):
        return (point - self.center) / self.radius

    def intersect(self, ray, light=False):
        # Move sphere to the origin and do the calculations
        ray_shifted = Ray(ray.start - self.center, ray.direction)
        a = np.dot(ray_shifted.direction, ray_shifted.direction)
        b = 2 * np.dot(ray_shifted.start, ray_shifted.direction)
        c = np.dot(ray_shifted.start, ray_shifted.start) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            # Does not intersect
            return None
        else:
            # Calculate t parameter and check if the sphere is behind the screen
            t_0 = (- b - discriminant ** 0.5) / (2 * a)
            t_1 = (- b + discriminant ** 0.5) / (2 * a)

            if light:
                # Ray needs to intersect with the object on a point between the light
                # source and the ray's starting point. So we check if the t value is
                # between 0 and 1.
                if THRESHOLD < t_0 < 1:
                    return t_0
                elif THRESHOLD < t_1 < 1:
                    return t_1
                else:
                    return None
            else:
                # t value needs to be greater than 1 because we don't want to show
                # the objects between the screen and the eye.
                if t_0 > 1:
                    return t_0
                elif t_1 > 1:
                    return t_1
                else:
                    return None


class Scene:
    def __init__(self):
        # Initialize parameters
        self.surfaces = []
        self.eye = np.array([0, 0, 0])
        self.shadow_coefficient = 0.1
        self.light_position = np.array([500, 500, 500])
        self.image = np.zeros((1000, 1000, 3))
        self.ambient_coefficient = 0.1
        self.ambient = np.array([0, 0, 0])
        self.bg_color = np.array([0, 0, 0])

    def get_input(self, filename):
        """
        :param filename: file with input
        Input format
        Ambient
        n: Number of spheres
        next n rows: color of sphere i
                     position of sphere i
                     radius of sphere i
                     diffuse coefficient of sphere i
        n: Number of planes
        next n rows: color of plane i
                     point on plane i
                     normal of plane i
                     diffuse coefficient of plane i
        :return: None
        """
        f = open(filename, 'r')
        self.ambient = np.array(list(map(int, f.readline().split())))
        sphere_num = int(f.readline())

        for _ in range(sphere_num):
            color = list(map(int, f.readline().split()))
            position = list(map(int, f.readline().split()))
            radius = int(f.readline())
            diffuse_coefficient = float(f.readline())
            reflection_coefficient = float(f.readline())
            s = Sphere(position, radius, color, diffuse_coefficient, reflection_coefficient)
            self.surfaces.append(s)
        plane_num = int(f.readline())
        for _ in range(plane_num):
            color = list(map(int, f.readline().split()))
            point = list(map(int, f.readline().split()))
            normal = list(map(float, f.readline().split()))
            diffuse_coefficient = float(f.readline())
            reflection_coefficient = float(f.readline())
            s = Plane(point, normal, color, diffuse_coefficient, reflection_coefficient)
            self.surfaces.append(s)
        f.close()

    def trace(self):
        for i in range(1000):
            for j in range(1000):
                t_min = float('inf')
                surface_min = None
                p_x = (i / 10) - 50 + 0.05
                p_y = ((999 - j) / 10) - 50 + 0.05
                p_z = 100
                ray = self._generate_ray(p_x, p_y, p_z)
                for surface in self.surfaces:
                    t = surface.intersect(ray)
                    if t is not None:
                        if surface_min is None or t < t_min:
                            t_min = t
                            surface_min = surface

                if surface_min is not None:
                    illumination = self._calculate_illumination(ray, surface_min, t_min, 0)
                    self.image[j][i] = illumination
                    for k in range(3):
                        self.image[j][i][k] = min(self.image[j][i][k], 255)
                else:
                    self.image[j][i] = self.bg_color

    def calculate_reflection_color(self, ray, depth):
        surface_min = None
        t_min = None
        for s in self.surfaces:
            t = s.intersect(ray)
            if t is not None and (surface_min is None or t < t_min):
                surface_min = s
                t_min = t
        if t_min is None:
            return self.bg_color

        col = self._calculate_illumination(ray, surface_min, t_min, depth) / depth
        return col

    def _calculate_illumination(self, ray, surface, t, depth):
        intersection_point = ray.direction * t + ray.start
        normal = surface.get_surface_normal(intersection_point)
        light_ray = Ray(intersection_point, self.light_position - intersection_point)

        see_light = True
        # Shadow
        for s in self.surfaces:
            t = s.intersect(light_ray, True)
            if t is not None and s != surface:
                see_light = False
                break
        # Ambient
        ambient = self.ambient * self.ambient_coefficient
        # diffuse color
        dot = np.dot(normalize(light_ray.direction), normal)
        diffuse = surface.color * surface.diffuse_coefficient * max(0, dot)
        color = ambient + diffuse * see_light

        # Reflection
        if depth < REFLECTION_DEPTH:
            dir = normalize(ray.direction - normal * 2 * np.dot(ray.direction, normal))
            start = intersection_point + THRESHOLD * normal
            reflection_ray = Ray(start, dir)
            color += (surface.reflection_coefficient * self.calculate_reflection_color(reflection_ray, depth + 1))
        return color

    def _generate_ray(self, x, y, z):
        # divisor = ((x - self.eye[0])**2 + (y - self.eye[1])**2 + (z - self.eye[2])**2)**0.5
        # Direction does not have to be a unit vector
        direction = np.array([x, y, z]) - self.eye
        ray = Ray(self.eye, direction)
        return ray

    def render(self, show=False):
        # show scene
        self.trace()
        if show:
            plt.imshow(self.image.astype('uint8'))
            plt.show()

        im = Image.fromarray(self.image.astype('uint8'))
        im.save('scene.jpg')


def main():
    scene = Scene()
    scene.get_input(argv[1])
    scene.render(show=True)


if __name__ == "__main__":
    main()
