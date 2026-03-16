interface NowPlayingProps {
  title: string;
  bpm: number;
  musicalKey: string;
  position: number;
  duration: number;
  isPlaying: boolean;
}

export function NowPlaying({ title, bpm, musicalKey, position, duration, isPlaying }: NowPlayingProps) {
  const progress = duration > 0 ? (position / duration) * 100 : 0;
  const formatTime = (s: number) => {
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${min}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="now-playing">
      <div className="np-status">{isPlaying ? "Now Playing" : "Paused"}</div>
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
