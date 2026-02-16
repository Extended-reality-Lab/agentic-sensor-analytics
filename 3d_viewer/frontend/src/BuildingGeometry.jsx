import React, { useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import * as THREE from 'three';

export default function BuildingGeometry({ modelUrl }) {
  // If model URL provided, load it
  const gltf = modelUrl ? useLoader(GLTFLoader, modelUrl) : null;

  // Otherwise create procedural box building
  const proceduralBuilding = useMemo(() => {
    if (modelUrl) return null;

    const buildingWidth = 40;
    const buildingDepth = 20;
    const floorHeight = 4;
    const numFloors = 3;

    return (
      <group>
        {/* Main building box */}
        <mesh position={[0, (numFloors * floorHeight) / 2, 0]} receiveShadow>
          <boxGeometry args={[buildingWidth, numFloors * floorHeight, buildingDepth]} />
          <meshStandardMaterial 
            color="#d4d4d4" 
            transparent 
            opacity={0.6}
            side={THREE.DoubleSide}
          />
        </mesh>

        {/* Floor separators */}
        {[1, 2, 3].map((floor) => (
          <mesh 
            key={floor} 
            position={[0, floor * floorHeight, 0]}
            rotation={[0, 0, 0]}
          >
            <boxGeometry args={[buildingWidth + 0.2, 0.1, buildingDepth + 0.2]} />
            <meshStandardMaterial color="#888888" />
          </mesh>
        ))}

        {/* Floor labels */}
        {[1, 2, 3].map((floor) => (
          <mesh 
            key={`label-${floor}`}
            position={[-buildingWidth/2 - 2, floor * floorHeight - 2, 0]}
          >
            <boxGeometry args={[1, 0.5, 0.1]} />
            <meshBasicMaterial color="#333333" />
          </mesh>
        ))}
      </group>
    );
  }, [modelUrl]);

  if (gltf) {
    return <primitive object={gltf.scene} />;
  }

  return proceduralBuilding;
}