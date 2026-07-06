import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from game import HokmGame, Phase

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"

rooms: dict[str, HokmGame] = {}
connections: dict[str, dict[int, WebSocket]] = {}  # room_id -> {user_id: ws}


def get_or_create_room(room_id: str) -> HokmGame:
    if room_id not in rooms:
        rooms[room_id] = HokmGame(room_id)
        connections[room_id] = {}
    return rooms[room_id]


async def broadcast(room_id: str):
    game = rooms.get(room_id)
    if not game:
        return
    for user_id, ws in list(connections.get(room_id, {}).items()):
        try:
            await ws.send_text(json.dumps({
                "type": "state",
                "state": game.public_state(viewer_user_id=user_id),
            }))
        except Exception:
            pass


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    game = get_or_create_room(room_id)
    user_id = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "join":
                user_id = int(msg["user_id"])
                name = msg.get("name", "بازیکن")[:24]
                seat = game.join(user_id, name)
                connections[room_id][user_id] = websocket
                if seat is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "میز پره! فقط ۴ نفر جا داره."}))
                else:
                    if game.all_seated() and game.phase == Phase.WAITING:
                        game.start_round()
                    await broadcast(room_id)

            elif action == "choose_hokm" and user_id is not None:
                seat = game.seat_of(user_id)
                if seat is not None:
                    game.choose_hokm(seat, msg.get("suit"))
                    await broadcast(room_id)

            elif action == "play_card" and user_id is not None:
                seat = game.seat_of(user_id)
                if seat is not None:
                    game.play_card(seat, msg.get("card"))
                    await broadcast(room_id)

            elif action == "next_round" and user_id is not None:
                game.continue_next_round()
                await broadcast(room_id)

    except WebSocketDisconnect:
        if user_id is not None:
            game.mark_disconnected(user_id)
            connections.get(room_id, {}).pop(user_id, None)
            await broadcast(room_id)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
