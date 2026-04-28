(() => {
  const timeSlices = [
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08"
  ];

  const clusterDefs = [
    { id: 0, name: "稳定贸易型", center: [-0.58, 0.52], color: "#35d0ff" },
    { id: 1, name: "高频批发型", center: [0.48, 0.55], color: "#7cff6b" },
    { id: 2, name: "跨群经纪型", center: [-0.34, -0.42], color: "#ffc857" },
    { id: 3, name: "异常波动型", center: [0.55, -0.35], color: "#ff5f7e" }
  ];

  const companyCount2d = 48;
  const embeddingData = [];
  for (let i = 0; i < companyCount2d; i += 1) {
    const cluster = clusterDefs[i % clusterDefs.length];
    const ring = Math.floor(i / clusterDefs.length);
    const angle = (i * 1.73) % (Math.PI * 2);
    const entryIndex = i % 13 === 0 ? 2 : i % 11 === 0 ? 1 : 0;
    const exitIndex = i % 17 === 0 ? 6 : timeSlices.length - 1;

    timeSlices.forEach((timeSlice, t) => {
      if (t < entryIndex || t > exitIndex) return;
      const driftX = Math.sin(t * 0.72 + i * 0.31) * 0.055 + (t - 3.5) * 0.01;
      const driftY = Math.cos(t * 0.58 + i * 0.27) * 0.055 + Math.sin(t * 0.34 + cluster.id) * 0.028;
      const localRadius = 0.045 + (ring % 5) * 0.016;
      const embRelation = clamp(cluster.center[0] + Math.cos(angle + t * 0.08) * localRadius + driftX, -0.95, 0.95);
      const embFeature = clamp(cluster.center[1] + Math.sin(angle + t * 0.1) * localRadius + driftY, -0.95, 0.95);

      embeddingData.push({
        company_id: `C${String(i + 1).padStart(3, "0")}`,
        time_slice: timeSlice,
        emb_relation: round(embRelation, 4),
        emb_feature: round(embFeature, 4),
        cluster_id: cluster.id,
        cluster_name: cluster.name
      });
    });
  }

  const companyCount3d = 96;
  const baseCompanies = Array.from({ length: companyCount3d }, (_, i) => {
    const group = i % 8;
    const angle = (i / companyCount3d) * Math.PI * 2;
    const groupCenters = [
      [12, 16],
      [34, 12],
      [62, 16],
      [86, 24],
      [18, 50],
      [48, 52],
      [78, 54],
      [58, 84]
    ];
    const radius = 5 + (i % 7) * 1.55;
    return {
      id: `N${String(i + 1).padStart(3, "0")}`,
      group,
      x: Math.round(groupCenters[group][0] + Math.cos(angle * 3.1) * radius),
      y: Math.round(groupCenters[group][1] + Math.sin(angle * 2.7) * radius)
    };
  });

  const relationPairs = [];
  const pairSet = new Set();
  const addPair = (a, b) => {
    const source = Math.min(a, b);
    const target = Math.max(a, b);
    const key = `${source}-${target}`;
    if (source !== target && !pairSet.has(key)) {
      pairSet.add(key);
      relationPairs.push([source, target]);
    }
  };
  for (let i = 0; i < companyCount3d; i += 1) {
    [1, 2, 3, 5, 8, 13, 21].forEach((offset) => {
      if ((i + offset) % 3 !== 0 || offset <= 5) addPair(i, (i + offset) % companyCount3d);
    });
    if (i % 4 === 0) addPair(i, (i + 34) % companyCount3d);
    if (i % 6 === 0) addPair(i, (i + 47) % companyCount3d);
  }

  const terrainData = [];
  timeSlices.forEach((timeSlice, t) => {
    relationPairs.forEach(([sourceIndex, targetIndex], pairIndex) => {
      const source = companyAtTime(baseCompanies[sourceIndex], t, sourceIndex);
      const target = companyAtTime(baseCompanies[targetIndex], t, targetIndex);
      const dx = source.x - target.x;
      const dy = source.y - target.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const wave = Math.sin(t * 1.18 + pairIndex * 0.73);
      const shock = Math.cos(t * 0.91 + sourceIndex * 0.41 + targetIndex * 0.19);
      const strength = clamp(0.5 + wave * 0.48 + (100 - distance) / 220 + (pairIndex % 11 === 0 ? 0.26 : -0.12), 0.005, 1);
      const confidenceSignal = Math.sin(t * 1.45 + pairIndex * 0.71) + Math.cos(targetIndex * 0.43 - t * 0.92) * 1.05;
      const confidence = clamp(0.5 + Math.tanh(confidenceSignal * 3.1) * 0.498 + (pairIndex % 17 === 0 ? -0.28 : 0), 0.001, 0.999);
      const spike = pairIndex % 19 === 0 ? 58 : pairIndex % 23 === 0 ? -52 : pairIndex % 29 === 0 ? 38 : 0;
      const embedding = clamp(2 + strength * 118 + confidence * 28 + shock * 32 + spike, 0, 180);

      terrainData.push({
        company_id: source.id,
        related_company_id: target.id,
        time_slice: timeSlice,
        pos_x: round(source.x, 3),
        pos_y: round(source.y, 3),
        rel_embedding: round(embedding, 3),
        confidence: round(confidence, 3),
        relation_strength: round(strength, 3)
      });
    });
  });

  window.Q1MockData = {
    timeSlices,
    clusterDefs,
    embeddingData,
    terrainData
  };

  function companyAtTime(company, t, index) {
    return {
      id: company.id,
      group: company.group,
      x: clamp(company.x, 2, 98),
      y: clamp(company.y, 2, 98)
    };
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function round(value, digits) {
    const scale = 10 ** digits;
    return Math.round(value * scale) / scale;
  }
})();
