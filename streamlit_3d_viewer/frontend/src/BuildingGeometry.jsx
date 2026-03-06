import React, { Suspense, useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

function LoadedModel({ modelUrl }) {
  const { scene } = useGLTF(modelUrl);

  const { scale, position } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const s = 15 / size.y;
    return {
      scale: s,
      position: [
        -center.x * s,
        -box.min.y * s,
        -center.z * s
      ]
    };
  }, [scene]);

  return (
    <group scale={[scale, scale, scale]} position={position}>
      <primitive object={scene} />
    </group>
  );
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
    useGLTF.preload(modelUrl);
    return (
      <Suspense fallback={null}>
        <LoadedModel modelUrl={modelUrl} />
      </Suspense>
    );
  }

  return <ProceduralBuilding />;
}