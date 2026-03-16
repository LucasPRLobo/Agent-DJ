interface QueueTrack {
  title: string;
  bpm: number;
  key: string;
}

interface QueueProps {
  tracks: QueueTrack[];
  currentIndex: number;
}

export function Queue({ tracks, currentIndex }: QueueProps) {
  const upcoming = tracks.slice(currentIndex + 1, currentIndex + 4);

  return (
    <div className="queue">
      <h3>Up Next</h3>
      {upcoming.length === 0 ? (
        <p className="queue-empty">No tracks queued</p>
      ) : (
        <ul>
          {upcoming.map((track, i) => (
            <li key={i} className="queue-item">
              <span className="queue-num">{i + 1}.</span>
              <span className="queue-title">{track.title}</span>
              <span className="queue-info">
                {track.bpm.toFixed(0)} BPM | {track.key}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
