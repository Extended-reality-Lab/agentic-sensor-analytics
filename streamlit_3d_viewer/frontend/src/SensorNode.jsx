import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function SensorNode({ 
  nodeId, 
  position, 
  isActive, 
  isHovered,
  sensorTypes,
  room,
  onHover,
  onClick 
}) {
  const groupRef = useRef();
  const meshRef = useRef();
  const [pulsePhase, setPulsePhase] = useState(0);

  // Animate active nodes with pulse effect
  useFrame((state, delta) => {
    if (!meshRef.current || !groupRef.current) return;

    // Group handles hover scale, mesh handles pulse
    groupRef.current.scale.setScalar(isHovered && !isActive ? 1.3 : 1);

    if (isActive) {
      setPulsePhase((prev) => prev + delta * 2);
      const pulse = 1 + Math.sin(pulsePhase) * 0.2;
      meshRef.current.scale.setScalar(pulse);
    } else {
      meshRef.current.scale.setScalar(1);
    }
  });

  // Color based on state
  const getColor = () => {
    if (isActive) return '#00ff00';
    if (isHovered) return '#ffff00';
    return '#0088ff';
  };

  return (
    <group ref={groupRef} position={position}>
      {/* Main sphere */}
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(nodeId);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          onHover(null);
          document.body.style.cursor = 'auto';
        }}
      >
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshStandardMaterial
          color={getColor()}
          emissive={getColor()}
          emissiveIntensity={isActive ? 0.5 : 0.3}
          metalness={0.5}
          roughness={0.2}
        />
      </mesh>

      {/* Glow ring effect for active node */}
      {isActive && (
        <mesh>
          <ringGeometry args={[0.5, 0.7, 32]} />
          <meshBasicMaterial 
            color="#00ff00" 
            transparent 
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}

      {/* Vertical line connecting to floor */}
      <mesh position={[0, -position[1] / 2, 0]}>
        <cylinderGeometry args={[0.02, 0.02, position[1], 8]} />
        <meshBasicMaterial color="#666666" transparent opacity={0.3} />
      </mesh>
    </group>
  );
}