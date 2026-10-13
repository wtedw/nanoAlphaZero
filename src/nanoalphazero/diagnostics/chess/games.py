"""Self-contained PGN viewer for notebooks and experiment dashboards."""

import io
import json

import chess
import chess.pgn
import chess.svg


def games_html(pgn_text: str) -> str:
    """Render all mainlines, retaining the complete original PGN for download.

    Python handles chess rules (including FEN starts, castling, en passant and
    promotions). The browser only displays positions; it needs no chess library,
    network connection, tracker, or model. Comments/variations remain in the PGN.
    """
    stream = io.StringIO(pgn_text)
    games = []
    while True:
        start = stream.tell()
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        if game.errors:
            raise ValueError(f"Cannot display game {len(games) + 1}: {game.errors[0]}")
        board = game.board()
        positions, moves = [board.fen()], []
        for move in game.mainline_moves():
            label = f"{board.fullmove_number}{'.' if board.turn else '...'} {board.san(move)}"
            moves.append(label)
            board.push(move)
            positions.append(board.fen())
        games.append(dict(headers=dict(game.headers), positions=positions, moves=moves,
                          pgn=pgn_text[start:stream.tell()]))
    pieces = {symbol: chess.svg.piece(chess.Piece.from_symbol(symbol))
              for symbol in "PNBRQKpnbrqk"}
    # PGN headers/comments are untrusted text, never executable HTML or JS.
    payload = json.dumps(dict(games=games, pgn=pgn_text, pieces=pieces), ensure_ascii=True).replace("<", "\\u003c")
    return _HTML.replace("__PAYLOAD__", payload)


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess games</title><style>
body{font:14px system-ui,sans-serif;margin:12px;color:#222;background:#fff}
button,select{font:inherit;padding:6px;margin:2px;cursor:pointer}select{max-width:100%}
.layout{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
#board{display:grid;grid-template-columns:repeat(8,1fr);width:min(360px,90vw);aspect-ratio:1}
.square{position:relative;aspect-ratio:1;display:flex}.light{background:#eee0c5}.dark{background:#91a58b}
.square svg{width:100%;height:100%}.coord{position:absolute;bottom:1px;left:2px;font-size:9px;color:#333}
#moves{position:relative;max-height:320px;overflow:auto;max-width:420px}#moves button{border:0;background:#eee}
#moves button.active{background:#214f78;color:white}#status{margin:8px 0}
textarea{box-sizing:border-box;width:100%;height:160px;font:12px monospace;margin-top:8px}
details{margin-top:12px}button:disabled{cursor:default;opacity:.45}
</style></head><body><div class="chess-game-viewer">
<label>Game <select id="game" aria-label="Game"></select></label>
<button id="download-all">Download complete PGN</button>
<div class="layout"><div><div id="board" aria-label="Chessboard"></div>
<div><button id="first" aria-label="First position">|◀</button><button id="prev" aria-label="Previous move">◀</button>
<button id="next" aria-label="Next move">▶</button><button id="last" aria-label="Last position">▶|</button>
<button id="flip">Flip board</button></div><div id="status" aria-live="polite"></div></div>
<div><div id="description"></div><div id="moves"></div></div></div>
<details><summary>Selected game PGN</summary><button id="download-game">Download selected game</button>
<textarea id="selected-pgn" aria-label="Selected game PGN" readonly></textarea></details>
<details><summary>Complete PGN text</summary><textarea id="full-pgn" aria-label="Complete PGN" readonly></textarea></details>
<script id="game-data" type="application/json">__PAYLOAD__</script>
<script>
(()=>{
const root=document.currentScript.parentElement;
const el=id=>root.querySelector('#'+id);
const data=JSON.parse(el('game-data').textContent);
let gameIndex=0, ply=0, flipped=false;
el('full-pgn').value=data.pgn;
data.games.forEach((g,i)=>{
  const o=document.createElement('option');o.value=i;
  o.textContent=`${i+1}. ${g.headers.White || '?'} – ${g.headers.Black || '?'} (${g.headers.Result || '*'})`;
  el('game').append(o);
});
function render(){
  const g=data.games[gameIndex];if(!g)return;
  const fen=g.positions[ply], cells=[];
  for(const c of fen.split(' ')[0]){
    if(c==='/')continue;
    if(/[1-8]/.test(c))cells.push(...Array(Number(c)).fill(''));else cells.push(c);
  }
  el('board').replaceChildren();
  for(let n=0;n<64;n++){
    const i=flipped?63-n:n, r=Math.floor(i/8), c=i%8;
    const square=document.createElement('div');square.className='square '+((r+c)%2?'dark':'light');
    square.dataset.square='abcdefgh'[c]+(8-r);
    square.dataset.piece=cells[i];
    if(cells[i])square.innerHTML=data.pieces[cells[i]];
    const label=document.createElement('span');label.className='coord';label.textContent=square.dataset.square;
    square.append(label);el('board').append(square);
  }
  el('status').textContent=`Ply ${ply}/${g.moves.length} · ${fen.split(' ')[1]==='w'?'White':'Black'} to move`;
  el('first').disabled=el('prev').disabled=ply===0;
  el('last').disabled=el('next').disabled=ply===g.moves.length;
  for(const b of el('moves').children)b.classList.toggle('active',Number(b.dataset.ply)===ply);
  const list=el('moves'), active=list.querySelector('.active');
  if(active){
    if(active.offsetTop<list.scrollTop)list.scrollTop=active.offsetTop;
    else if(active.offsetTop+active.offsetHeight>list.scrollTop+list.clientHeight)
      list.scrollTop=active.offsetTop+active.offsetHeight-list.clientHeight;
  }
}
function selectGame(){
  gameIndex=Number(el('game').value);ply=0;
  const g=data.games[gameIndex];if(!g)return;
  el('selected-pgn').value=g.pgn;
  el('description').textContent=[g.headers.Event,g.headers.Result,g.headers.Termination].filter(Boolean).join(' · ');
  el('moves').replaceChildren();
  ['Start',...g.moves].forEach((label,i)=>{
    const b=document.createElement('button');b.textContent=label;b.dataset.ply=i;
    b.onclick=()=>{ply=i;render()};el('moves').append(b);
  });render();
}
function advance(delta){const g=data.games[gameIndex];if(g){ply=Math.max(0,Math.min(g.moves.length,ply+delta));render()}}
el('game').onchange=selectGame;
el('first').onclick=()=>{ply=0;render()};el('prev').onclick=()=>advance(-1);
el('next').onclick=()=>advance(1);el('last').onclick=()=>advance(data.games[gameIndex]?.moves.length || 0);
el('flip').onclick=()=>{flipped=!flipped;render()};
root.addEventListener('keydown',e=>{
  if(['TEXTAREA','SELECT','INPUT','BUTTON'].includes(e.target.tagName))return;
  if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();advance(e.key==='ArrowRight'?1:-1)}
});
function download(text,name){
  const url=URL.createObjectURL(new Blob([text],{type:'application/x-chess-pgn;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
el('download-all').onclick=()=>download(data.pgn,'games.pgn');
el('download-game').onclick=()=>download(data.games[gameIndex].pgn,`game-${gameIndex+1}.pgn`);
if(data.games.length)selectGame();else{
  el('status').textContent='No games in this PGN.';
  for(const id of ['game','first','prev','next','last','flip','download-game'])el(id).disabled=true;
}
})();
</script></div></body></html>'''
