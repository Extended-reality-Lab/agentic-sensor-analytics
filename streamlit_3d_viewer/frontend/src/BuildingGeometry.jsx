import React, { Suspense } from 'react';
import { useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import * as THREE from 'three';

function LoadedModel({ modelUrl }) {
  const gltf = useLoader(GLTFLoader, modelUrl);
  return <primitive object={gltf.scene} />;
}

function ProceduralBuilding() {
  const buildingWidth = 40;
  const buildingDepth = 20;
  const floorHeight = 4;
  const numFloors = 3;

  return (
    <group>
      <mesh position={[0, (numFloors * floorHeight) / 2, 0]} receiveShadow>
        <boxGeometry args={[buildingWidth, numFloors * floorHeight, buildingDepth]} />
        <meshStandardMaterial 
          color="#d4d4d4" 
          transparent 
          opacity={0.6}
          side={THREE.DoubleSide}
        />
      </mesh>

      {[1, 2, 3].map((floor) => (
        <mesh 
          key={floor} 
          position={[0, floor * floorHeight, 0]}
        >
          <boxGeometry args={[buildingWidth + 0.2, 0.1, buildingDepth + 0.2]} />
          <meshStandardMaterial color="#888888" />
        </mesh>
      ))}
    </group>
  );
}

export default function BuildingGeometry({ modelUrl }) {
  if (modelUrl) {
    return (
      <Suspense fallback={<ProceduralBuilding />}>
        <LoadedModel modelUrl={modelUrl} />
      </Suspense>
    );
  }

  return <ProceduralBuilding />;
}