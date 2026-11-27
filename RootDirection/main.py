from math import atan2, exp

import pygame


WIDTH, HEIGHT = 1120, 820
ARENA_CENTER_X = 480
ARENA_HALF_WIDTH = 160
GROUND_Y = 432
GROUND_VIEW_Y = 650
VIEW_SCALE = 2.5

RENDER_FPS = 60
PHYSICS_HZ = 120
PHYSICS_STEP = 1.0 / PHYSICS_HZ
MAX_FRAME_MS = 100

BACKGROUND = (240, 250, 250)
GROUND_COLOR = (220, 230, 230)
GROUND_LINE = (100, 140, 140)
DARK = (30, 60, 70)
SKIN = (240, 180, 140)
SHIRT = (50, 120, 120)
CONTROLS_COLOR = (60, 80, 90)

MOVE_SPEED = 230.0
JUMP_SPEED = 6450.0 / 13.0
GRAVITY = 1500.0
CHARACTER_HALF_WIDTH = 22
HEAD_OFFSET_Y = -74
HEAD_RADIUS = 20
SHOULDER_OFFSET_X = 12
SHOULDER_OFFSET_Y = -41
SHOULDER_RADIUS = 10

HAIR_STRANDS_PER_SIDE = 6
HAIR_PARTICLE_COUNT = 10
HAIR_SPRING_STIFFNESS = 5200.0
HAIR_SPRING_DAMPING = 26.0
HAIR_AIR_DRAG = 7.0
CONSTRAINT_ITERATIONS = 12
ROOT_DIRECTION_RATE = 28.0
ROOT_OUTWARD_TILT = 0.35
WIND_ANGLE_STEP = 15
WIND_STRENGTH_STEP = 100
MAX_WIND_STRENGTH = 1000
MIN_HAIR_AIR_DRAG = 4.0
MAX_HAIR_AIR_DRAG = 16.0
MIN_CONSTRAINT_ITERATIONS = 4
MAX_CONSTRAINT_ITERATIONS = 20
HAIR_MIN_DISTANCE = 1e-6
HAIR_PARTICLE_RADIUS = 1.7
HAIR_ROOT_ANGLES = (-168, -119)
HAIR_STRAND_LENGTHS = (60.0, 54.0)
HAIR_COLORS = ((70, 40, 40), (100, 60, 50), (130, 80, 60))


class HairSettings:
    def __init__(self):
        self.wind_angle = 0
        self.wind_strength = 0
        self.air_drag = HAIR_AIR_DRAG
        self.iterations = CONSTRAINT_ITERATIONS

    def WindForce(self):
        return pygame.Vector2(self.wind_strength, 0).rotate(self.wind_angle)

    def Adjust(self, key):
        if key in (pygame.K_q, pygame.K_e):
            change = WIND_ANGLE_STEP if key == pygame.K_e else -WIND_ANGLE_STEP
            self.wind_angle = (self.wind_angle + change) % 360
        elif key in (pygame.K_DOWN, pygame.K_UP):
            change = WIND_STRENGTH_STEP if key == pygame.K_UP else -WIND_STRENGTH_STEP
            self.wind_strength = max(0, min(MAX_WIND_STRENGTH, self.wind_strength + change))
        elif key in (pygame.K_z, pygame.K_x):
            change = 1 if key == pygame.K_x else -1
            self.air_drag = max(MIN_HAIR_AIR_DRAG, min(MAX_HAIR_AIR_DRAG, self.air_drag + change))
        elif key in (pygame.K_c, pygame.K_v):
            change = 2 if key == pygame.K_v else -2
            self.iterations = max(MIN_CONSTRAINT_ITERATIONS,
                                  min(MAX_CONSTRAINT_ITERATIONS, self.iterations + change))


class CircleCollider:
    def __init__(self, position, offset, radius):
        self.offset = pygame.Vector2(offset)
        self.center = position + self.offset
        self.previous_center = self.center.copy()
        self.velocity = pygame.Vector2()
        self.radius = radius

    def Update(self, position, velocity):
        self.previous_center = self.center
        self.center = position + self.offset
        self.velocity = velocity

    def SolvePosition(self, particle):
        if particle.fixed:
            return

        radius = self.radius + HAIR_PARTICLE_RADIUS
        offset = particle.position - self.center
        distance_squared = offset.length_squared()
        if distance_squared >= radius * radius:
            return

        if distance_squared < HAIR_MIN_DISTANCE ** 2:
            offset = particle.previous_position - self.previous_center
            if offset.length_squared() < HAIR_MIN_DISTANCE ** 2:
                offset.update(0, -1)

        particle.position = self.center + offset.normalize() * radius

    def SolveVelocity(self, particle):
        if particle.fixed:
            return

        radius = self.radius + HAIR_PARTICLE_RADIUS + HAIR_MIN_DISTANCE
        offset = particle.position - self.center
        distance_squared = offset.length_squared()
        if not HAIR_MIN_DISTANCE ** 2 <= distance_squared <= radius * radius:
            return

        normal = offset.normalize()
        inward_speed = (particle.velocity - self.velocity).dot(normal)
        if inward_speed < 0:
            particle.velocity -= normal * inward_speed


class Character:
    def __init__(self):
        self.position = pygame.Vector2(ARENA_CENTER_X, GROUND_Y)
        self.velocity = pygame.Vector2()
        self.on_ground = True
        self.colliders = (
            CircleCollider(self.position, (0, HEAD_OFFSET_Y), HEAD_RADIUS),
            CircleCollider(self.position, (-SHOULDER_OFFSET_X, SHOULDER_OFFSET_Y), SHOULDER_RADIUS),
            CircleCollider(self.position, (SHOULDER_OFFSET_X, SHOULDER_OFFSET_Y), SHOULDER_RADIUS),
        )

    def HeadPosition(self):
        return self.position + pygame.Vector2(0, HEAD_OFFSET_Y)

    def Update(self, move_direction, jump_pressed, dt):
        if dt <= 0:
            return

        previous_position = self.position.copy()
        self.velocity.x = move_direction * MOVE_SPEED
        if jump_pressed and self.on_ground:
            self.velocity.y = -JUMP_SPEED

        self.velocity.y += GRAVITY * dt
        self.position += self.velocity * dt

        left = ARENA_CENTER_X - ARENA_HALF_WIDTH + CHARACTER_HALF_WIDTH
        right = ARENA_CENTER_X + ARENA_HALF_WIDTH - CHARACTER_HALF_WIDTH
        self.position.x = max(left, min(self.position.x, right))
        if self.position.x == left or self.position.x == right:
            self.velocity.x = 0.0

        self.on_ground = self.position.y >= GROUND_Y
        if self.on_ground:
            self.position.y = GROUND_Y
            self.velocity.y = 0.0

        collider_velocity = (self.position - previous_position) / dt
        for collider in self.colliders:
            collider.Update(self.position, collider_velocity)


class Particle:
    def __init__(self, position, mass=1.0, fixed=False):
        if mass <= 0:
            raise ValueError("Particle mass must be positive")
        self.position = pygame.Vector2(position)
        self.previous_position = self.position.copy()
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()
        self.mass = mass
        self.fixed = fixed

    def InverseMass(self):
        return 0.0 if self.fixed else 1.0 / self.mass

    def AddForce(self, force):
        if not self.fixed:
            self.acceleration += force / self.mass

    def Update(self, dt):
        if not self.fixed:
            self.previous_position = self.position.copy()
            self.velocity += self.acceleration * dt
            self.position += self.velocity * dt


class HairStrand:
    def __init__(self, head_position, root_angle, rest_length, color):
        self.root_offset = pygame.Vector2(HEAD_RADIUS, 0).rotate(root_angle)
        self.rest_length = rest_length
        self.color = color

        root_position = head_position + self.root_offset
        side = -1 if self.root_offset.x < 0 else 1
        normal = self.root_offset / HEAD_RADIUS
        tangent = pygame.Vector2(-normal.y, normal.x) * side
        self.root_direction = (tangent + normal * ROOT_OUTWARD_TILT).normalize()
        self.initial_direction = pygame.Vector2(0.4 * side, 1).normalize()
        self.particles = [
            Particle(root_position + self.initial_direction * rest_length * index, fixed=index == 0)
            for index in range(HAIR_PARTICLE_COUNT)
        ]

    def ApplySpring(self, first, second):
        offset = second.position - first.position
        distance = offset.length()
        if distance < HAIR_MIN_DISTANCE:
            return

        axis = offset / distance
        extension = distance - self.rest_length
        relative_speed = (second.velocity - first.velocity).dot(axis)
        force = axis * (HAIR_SPRING_STIFFNESS * extension + HAIR_SPRING_DAMPING * relative_speed)
        first.AddForce(force)
        second.AddForce(-force)

    def SolveDistance(self, first, second):
        first_weight = first.InverseMass()
        second_weight = second.InverseMass()
        total_weight = first_weight + second_weight
        if total_weight == 0:
            return

        offset = second.position - first.position
        distance = offset.length()
        if distance >= HAIR_MIN_DISTANCE:
            axis = offset / distance
        else:
            previous_offset = second.previous_position - first.previous_position
            axis = self.initial_direction
            if previous_offset.length_squared() >= HAIR_MIN_DISTANCE ** 2:
                axis = previous_offset.normalize()

        correction = axis * ((distance - self.rest_length) / total_weight)
        first.position += correction * first_weight
        second.position -= correction * second_weight

    def SolveRootDirection(self, root, first, weight):
        edge = first.position - root.position
        if first.fixed or edge.length_squared() < HAIR_MIN_DISTANCE ** 2:
            return

        direction = self.root_direction
        angle_error = atan2(edge.cross(direction), edge.dot(direction))
        first.position = root.position + edge.rotate_rad(angle_error * weight)

    def Update(self, head_position, dt, colliders=(), wind_force=None,
               air_drag=HAIR_AIR_DRAG, iterations=CONSTRAINT_ITERATIONS):
        if dt <= 0:
            return

        root = self.particles[0]
        root_position = head_position + self.root_offset
        root.previous_position = root.position
        root.velocity = (root_position - root.position) / dt
        root.position = root_position

        free_particles = self.particles[1:]
        links = list(zip(self.particles, free_particles))
        for particle in free_particles:
            particle.acceleration.update(0, GRAVITY)
            particle.acceleration -= particle.velocity * air_drag
            if wind_force is not None:
                particle.AddForce(wind_force)

        for first, second in links:
            self.ApplySpring(first, second)

        for particle in free_particles:
            particle.Update(dt)

        root_direction_weight = 1.0 - exp(-ROOT_DIRECTION_RATE * dt / max(1, iterations))
        for _ in range(iterations):
            self.SolveRootDirection(root, free_particles[0], root_direction_weight)
            for first, second in links:
                self.SolveDistance(first, second)
            for particle in free_particles:
                for collider in colliders:
                    collider.SolvePosition(particle)

        for particle in free_particles:
            particle.velocity = (particle.position - particle.previous_position) / dt
            for collider in colliders:
                collider.SolveVelocity(particle)


def Project(position):
    x = WIDTH / 2 + (position.x - ARENA_CENTER_X) * VIEW_SCALE
    y = GROUND_VIEW_Y + (position.y - GROUND_Y) * VIEW_SCALE
    return round(x), round(y)


def ProjectRect(position, left, top, width, height):
    x, y = Project(position + (left, top))
    return pygame.Rect(x, y, round(width * VIEW_SCALE), round(height * VIEW_SCALE))


def DrawHair(window, hair_strands):
    radius = max(1, round(HAIR_PARTICLE_RADIUS * VIEW_SCALE))
    line_width = max(1, round(2 * radius * 0.4))
    for strand in hair_strands:
        points = [Project(particle.position) for particle in strand.particles]
        pygame.draw.lines(window, strand.color, False, points, line_width)
        for point in points:
            pygame.draw.circle(window, strand.color, point, radius)


def DrawSettings(window, font, settings):
    pygame.draw.line(window, GROUND_LINE, (30, 720), (WIDTH - 30, 720))
    wind_value = f"Wind {settings.wind_strength} / {MAX_WIND_STRENGTH}   |   {settings.wind_angle:03d}\N{DEGREE SIGN}"
    wind_rect = window.blit(font.render(wind_value, True, CONTROLS_COLOR), (30, 740))
    columns = (
        (460, f"Damping {settings.air_drag:g} / s"),
        (800, f"Solver iterations {settings.iterations}"),
    )
    for x, value in columns:
        window.blit(font.render(value, True, CONTROLS_COLOR), (x, 740))

    direction = pygame.Vector2(1, 0).rotate(settings.wind_angle)
    length = 20 + 10 * settings.wind_strength / MAX_WIND_STRENGTH
    tip = direction * length
    arrowhead_width = 10
    side = direction.rotate(90) * (arrowhead_width / 2)
    base = tip - direction * 10
    points = (pygame.Vector2(), tip, base + side, base - side)
    left = min(point.x for point in points)
    top = min(point.y for point in points)
    bottom = max(point.y for point in points)
    origin = pygame.Vector2(wind_rect.right + 10 - left, wind_rect.centery - (top + bottom) / 2)
    color = DARK if settings.wind_strength else GROUND_LINE
    pygame.draw.line(window, color, origin, origin + tip)
    pygame.draw.polygon(window, color, [origin + point for point in points[1:]])


def DrawScene(window, character, hair_strands, controls, font, settings):
    window.fill(BACKGROUND)
    floor_width = round(2 * ARENA_HALF_WIDTH * VIEW_SCALE)
    ground = pygame.Rect((WIDTH - floor_width) // 2, GROUND_VIEW_Y, floor_width, 60)
    pygame.draw.rect(window, GROUND_COLOR, ground)
    pygame.draw.line(window, GROUND_LINE, ground.topleft, (ground.right - 1, ground.top))

    position = character.position
    foot_x, foot_y = Project(position)
    for direction in (-1, 1):
        start = (foot_x + direction * 20, foot_y - 60)
        end = (foot_x + direction * 30, foot_y - 10)
        pygame.draw.line(window, DARK, start, end, 20)
        shoe = pygame.Rect(end[0] - 20, end[1], 40, 10)
        pygame.draw.rect(window, DARK, shoe)

    neck = pygame.Rect(foot_x - 10, foot_y - 150, 20, 30)
    body_top = SHOULDER_OFFSET_Y - SHOULDER_RADIUS
    body = ProjectRect(position, -CHARACTER_HALF_WIDTH, body_top, CHARACTER_HALF_WIDTH * 2, 30)
    pygame.draw.rect(window, SKIN, neck)
    pygame.draw.rect(window, SHIRT, body, border_radius=20)
    DrawHair(window, hair_strands)
    pygame.draw.circle(window, SKIN, Project(character.HeadPosition()), round(HEAD_RADIUS * VIEW_SCALE))

    for text, position in controls:
        window.blit(text, position)
    DrawSettings(window, font, settings)
    pygame.display.update()


def CreateScene():
    if HAIR_PARTICLE_COUNT < 2 or HAIR_STRANDS_PER_SIDE < 1:
        raise ValueError("Hair needs at least one strand per side and two particles per strand")

    character = Character()
    head_position = character.HeadPosition()
    hair_strands = []
    first_angle, last_angle = HAIR_ROOT_ANGLES
    longest, shortest = HAIR_STRAND_LENGTHS
    angle_divisor = max(1, HAIR_STRANDS_PER_SIDE - 1)

    for index in range(HAIR_STRANDS_PER_SIDE):
        fraction = index / angle_divisor
        left_angle = first_angle + (last_angle - first_angle) * fraction
        right_angle = -180 - left_angle
        strand_length = longest + (shortest - longest) * fraction
        rest_length = strand_length / (HAIR_PARTICLE_COUNT - 1)
        color = HAIR_COLORS[index % len(HAIR_COLORS)]
        for angle in (left_angle, right_angle):
            hair_strands.append(HairStrand(head_position, angle, rest_length, color))

    return character, hair_strands


def gameLoop():
    pygame.display.init()
    pygame.font.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("")
    controls_font = pygame.font.Font(None, 30)
    controls = [
        (controls_font.render(text, True, CONTROLS_COLOR), position)
        for text, position in (
            ("A / D : Move  Space : Jump  R : Reset ", (30, 30)),
            (f"Q / E Rotate     Up / Down +{WIND_STRENGTH_STEP} / -{WIND_STRENGTH_STEP}", (30, 770)),
            ("Z / X Decrease / Increase", (460, 770)),
            ("C / V Decrease / Increase", (800, 770)),
        )
    ]

    clock = pygame.time.Clock()
    character, hair_strands = CreateScene()
    settings = HairSettings()
    accumulator = 0
    jump_pressed = False

    while True:
        frame_ms = min(clock.tick(RENDER_FPS), MAX_FRAME_MS)
        accumulator += frame_ms * PHYSICS_HZ

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    jump_pressed = True
                elif event.key == pygame.K_r:
                    character, hair_strands = CreateScene()
                    settings = HairSettings()
                    accumulator = 0
                    jump_pressed = False
                else:
                    settings.Adjust(event.key)

        keys = pygame.key.get_pressed()
        move_left = keys[pygame.K_a]
        move_right = keys[pygame.K_d]
        move_direction = int(move_right) - int(move_left)
        wind_force = settings.WindForce() if settings.wind_strength else None

        steps, accumulator = divmod(accumulator, 1000)
        for _ in range(steps):
            character.Update(move_direction, jump_pressed, PHYSICS_STEP)
            head_position = character.HeadPosition()
            for strand in hair_strands:
                strand.Update(head_position, PHYSICS_STEP, character.colliders,
                              wind_force, settings.air_drag, settings.iterations)
            jump_pressed = False

        DrawScene(window, character, hair_strands, controls, controls_font, settings)


if __name__ == "__main__":
    gameLoop()
