"""
Confetti particle system for winning screens
"""
import cv2
import numpy as np
import random
import math


class ConfettiParticle:
    """Represents a single confetti particle"""
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Random velocity
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-8, -2)  # Upward initial velocity
        
        # Random rotation
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)
        
        # Random color (bright, festive colors)
        colors = [
            (0, 255, 255),    # Yellow
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (128, 0, 255),    # Purple
            (255, 165, 0),    # Orange
        ]
        self.color = random.choice(colors)
        
        # Gravity
        self.gravity = 0.3
        
        # Life (for fade out effect)
        self.life = 1.0
        self.life_decay = random.uniform(0.002, 0.005)
        
        # Shape type (0 = rectangle, 1 = circle)
        self.shape_type = random.choice([0, 1])
        
    def update(self):
        """Update particle position and physics"""
        # Apply gravity
        self.vy += self.gravity
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Update rotation
        self.rotation += self.rotation_speed
        
        # Update life
        self.life -= self.life_decay
        
        # Add some air resistance
        self.vx *= 0.99
        self.vy *= 0.99
        
    def is_alive(self):
        """Check if particle is still alive"""
        return self.life > 0
    
    def draw(self, screen):
        """Draw the particle on the screen"""
        if not self.is_alive():
            return
        
        # Calculate alpha based on life
        alpha = max(0, min(1, self.life))
        
        # Get color with alpha
        color = tuple(int(c * alpha) for c in self.color)
        
        # Draw based on shape type
        if self.shape_type == 0:
            # Rectangle confetti
            points = np.array([
                [-self.width/2, -self.height/2],
                [self.width/2, -self.height/2],
                [self.width/2, self.height/2],
                [-self.width/2, self.height/2]
            ], dtype=np.float32)
            
            # Rotate points
            angle_rad = math.radians(self.rotation)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            rotated_points = []
            for px, py in points:
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                rotated_points.append([int(self.x + rx), int(self.y + ry)])
            
            # Draw filled rectangle
            pts = np.array(rotated_points, dtype=np.int32)
            cv2.fillPoly(screen, [pts], color)
        else:
            # Circle confetti
            radius = max(self.width, self.height) // 2
            cv2.circle(screen, (int(self.x), int(self.y)), radius, color, -1)


class ConfettiSystem:
    """Manages a system of confetti particles"""
    
    def __init__(self, screen_width, screen_height, num_particles=150):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.particles = []
        self.num_particles = num_particles
        self.active = False
        
    def start(self, burst_x=None, burst_y=None, multiple_bursts=True, num_burst_points=2):
        """
        Start the confetti animation
        
        Args:
            burst_x: Single burst X position (if None, uses multiple bursts)
            burst_y: Single burst Y position (if None, uses multiple bursts)
            multiple_bursts: If True, create bursts from multiple points across the screen
            num_burst_points: Number of burst points to create (if multiple_bursts is True)
        """
        self.active = True
        self.particles = []
        
        if multiple_bursts and burst_x is None and burst_y is None:
            # Create multiple burst points across the screen
            burst_points = []
            margin = 100  # Margin from edges
            
            # Distribute burst points across the screen
            for i in range(num_burst_points):
                # Create points in a grid-like pattern with some randomness
                if num_burst_points == 1:
                    x = self.screen_width // 2
                    y = self.screen_height // 4
                else:
                    # Distribute horizontally and vertically
                    cols = int(num_burst_points ** 0.5) + 1
                    row = i // cols
                    col = i % cols
                    x_spacing = (self.screen_width - 2 * margin) / max(cols - 1, 1)
                    y_spacing = (self.screen_height - 2 * margin) / max((num_burst_points // cols), 1)
                    x = margin + col * x_spacing + random.uniform(-50, 50)
                    y = margin + row * y_spacing + random.uniform(-50, 50)
                    # Keep within bounds
                    x = max(margin, min(self.screen_width - margin, x))
                    y = max(margin, min(self.screen_height - margin, y))
                burst_points.append((x, y))
        else:
            # Single burst point
            if burst_x is None:
                burst_x = self.screen_width // 2
            if burst_y is None:
                burst_y = self.screen_height // 4
            burst_points = [(burst_x, burst_y)]
        
        # Create particles from each burst point
        particles_per_burst = self.num_particles // len(burst_points)
        remaining_particles = self.num_particles % len(burst_points)
        
        for i, (bx, by) in enumerate(burst_points):
            # Distribute remaining particles to first few bursts
            num_for_this_burst = particles_per_burst + (1 if i < remaining_particles else 0)
            
            for _ in range(num_for_this_burst):
                # Random starting position around burst point
                offset_x = random.uniform(-50, 50)
                offset_y = random.uniform(-50, 50)
                
                # Random size
                size = random.randint(8, 20)
                
                particle = ConfettiParticle(
                    bx + offset_x,
                    by + offset_y,
                    size,
                    size
                )
                self.particles.append(particle)
    
    def stop(self):
        """Stop creating new particles"""
        self.active = False
    
    def update(self):
        """Update all particles"""
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]
        
        # Update all particles
        for particle in self.particles:
            particle.update()
            
            # Wrap around screen edges (optional, for continuous effect)
            if particle.x < 0:
                particle.x = self.screen_width
            elif particle.x > self.screen_width:
                particle.x = 0
                
            if particle.y > self.screen_height:
                # Reset particle to top with new random position
                particle.y = random.uniform(-50, 0)
                particle.x = random.uniform(0, self.screen_width)
                particle.vy = random.uniform(-8, -2)
                particle.life = 1.0
    
    def draw(self, screen):
        """Draw all particles on the screen"""
        for particle in self.particles:
            particle.draw(screen)
    
    def has_particles(self):
        """Check if there are any active particles"""
        return len(self.particles) > 0


def create_confetti_overlay(screen, confetti_system, alpha=0.7):
    """
    Create an overlay with confetti particles
    
    Args:
        screen: The screen to draw on
        confetti_system: ConfettiSystem instance
        alpha: Transparency of confetti overlay (0-1)
    
    Returns:
        Screen with confetti drawn
    """
    # Create a copy of the screen
    overlay = screen.copy()
    
    # Draw confetti
    confetti_system.draw(overlay)
    
    # Blend with original screen
    result = cv2.addWeighted(screen, 1 - alpha, overlay, alpha, 0)
    
    return result

