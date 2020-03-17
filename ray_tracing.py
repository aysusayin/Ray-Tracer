import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)


class Ray:
    def __init__(self, start, direction):
        self.start = start
        self.direction = direction


class Scene:
    def __init__(self):
        # Initialize parameters
        self.sphere_num = 0
        self.spheres = []
        self.eye = np.array([0, 0, 0])
        self.shadow_coefficient = 0.1
        self.light_position = np.array([500, 500, 500])
        self.image = np.zeros((1000, 1000, 3))

    def get_input(self):
        # Get scene inputs
        self.sphere_num = int(input("Enter the number of spheres: "))
        self.spheres = []
        for i in range(1, self.sphere_num + 1):
            color = input("Color of the sphere %d in R G B format: " % i)
            color = list(map(int, color.split()))
            position = input("Position (x, y, z) of the sphere %d: " % i)
            position = list(map(int, position.split()))
            radius = int(input("Radius of the sphere %d: " % i))
            print("----------------------------------")
            s = Sphere(position, radius, color)
            self.spheres.append(s)

    def check_inputs(self):
        for s in self.spheres:
            print("Sphere color: ")
            print(s.color)
            print("position: ")
            print(s.position)
            print("radius: %d" % s.radius)
            print("----------------------------------------")

    def trace(self):
        for i in range(1000):
            for j in range(1000):
                t_min = float('inf')
                sphere_min = None
                p_x = (i / 10) - 50 + 0.05
                p_y = ((999-j) / 10) - 50 + 0.05
                p_z = 100
                ray = self._generate_ray(p_x, p_y, p_z)
                for sphere in self.spheres:
                    t = self._intersection(ray, sphere)
                    if t is not None:
                        if sphere_min is None or t < t_min:
                            t_min = t
                            sphere_min = sphere

                if sphere_min is not None:
                    illumination = self._calculate_illumination(ray, sphere_min, t_min)
                    self.image[j][i] = illumination

    def _calculate_illumination(self, ray, sphere, t):
        intersection_point = ray.direction * t + ray.start
        light_ray = Ray(intersection_point, self.light_position - intersection_point)
        for s in self.spheres:
            t = self._intersection(light_ray, s, True)
            if t is not None:
                return sphere.color * self.shadow_coefficient
        return sphere.color

    def _generate_ray(self, x, y, z):
        # divisor = ((x - self.eye[0])**2 + (y - self.eye[1])**2 + (z - self.eye[2])**2)**0.5
        # Direction does not have to be a unit vector
        direction = np.array([x, y, z]) - self.eye
        ray = Ray(self.eye, direction)
        return ray

    def _intersection(self, ray, sphere, light=False):
        # Move sphere to the origin and do the calculations
        ray_shifted = Ray(ray.start - sphere.center, ray.direction)
        a = np.dot(ray_shifted.direction, ray_shifted.direction)
        b = 2 * np.dot(ray_shifted.start, ray_shifted.direction)
        c = np.dot(ray_shifted.start, ray_shifted.start) - sphere.radius ** 2
        discriminant = b**2 - 4*a*c

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
                if 0.0001 < t_0 < 1:
                    return t_0
                elif 0.0001 < t_1 < 1:
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
    scene.get_input()
    scene.render()


if __name__ == "__main__":
    print("Welcome")
    main()
