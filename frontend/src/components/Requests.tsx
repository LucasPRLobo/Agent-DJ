interface SongRequest {
  query: string;
  from: string;
  votes: number;
}

interface RequestsProps {
  requests: SongRequest[];
  onVote?: (index: number) => void;
  canVote?: boolean;
}

export function Requests({ requests, onVote, canVote }: RequestsProps) {
  const sorted = [...requests].sort((a, b) => b.votes - a.votes);

  return (
    <div className="requests">
      <h3>Requests</h3>
      {sorted.length === 0 ? (
        <p className="requests-empty">No requests yet</p>
      ) : (
        <ul>
          {sorted.map((req, i) => (
            <li key={i} className="request-item">
              <span className="request-song">{req.query}</span>
              <span className="request-from">by {req.from}</span>
              <span className="request-votes">
                {canVote && onVote ? (
                  <button onClick={() => onVote(i)} className="vote-btn">
                    +{req.votes}
                  </button>
                ) : (
                  `+${req.votes}`
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
