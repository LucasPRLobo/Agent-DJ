interface NowPlayingProps {
  title: string;
  bpm: number;
  musicalKey: string;
  position: number;
  duration: number;
  isPlaying: boolean;
  onPause?: () => void;
  onResume?: () => void;
  onSkip?: () => void;
}

export function NowPlaying({ title, bpm, musicalKey, position, duration, isPlaying, onPause, onResume, onSkip }: NowPlayingProps) {
  const progress = duration > 0 ? (position / duration) * 100 : 0;
  const formatTime = (s: number) => {
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${min}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="now-playing">
      <div className="np-header">
        <div className="np-status">{isPlaying ? "Now Playing" : "Paused"}</div>
        <div className="np-controls">
          {isPlaying ? (
            <button className="np-btn" onClick={onPause} title="Pause">⏸</button>
          ) : (
            <button className="np-btn" onClick={onResume} title="Play">▶</button>
          )}
          <button className="np-btn" onClick={onSkip} title="Skip">⏭</button>
        </div>
      </div>
      <div className="np-title">{title || "No track loaded"}</div>
      <div className="np-info">
        {bpm > 0 && <span>{bpm.toFixed(1)} BPM</span>}
        {musicalKey && <span>{musicalKey}</span>}
      </div>
      <div className="np-progress-bar">
        <div className="np-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="np-time">
        <span>{formatTime(position)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
