interface EnergyArcProps {
  arc: number[];
  currentPosition: number; // 0-1
}

export function EnergyArc({ arc, currentPosition }: EnergyArcProps) {
  if (!arc || arc.length === 0) return null;

  const width = 300;
  const height = 60;
  const padding = 4;
  const barWidth = (width - padding * 2) / arc.length;
  const currentIdx = Math.floor(currentPosition * arc.length);

  return (
    <div className="energy-arc">
      <h3>Energy Arc</h3>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {arc.map((energy, i) => {
          const barHeight = energy * (height - padding * 2);
          const x = padding + i * barWidth;
          const y = height - padding - barHeight;
          const isCurrent = i === currentIdx;

          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={Math.max(barWidth - 1, 1)}
              height={barHeight}
              fill={isCurrent ? "#ff6b35" : i < currentIdx ? "#666" : "#4a9eff"}
              rx={1}
            />
          );
        })}
      </svg>
    </div>
  );
}
