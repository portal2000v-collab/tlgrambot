(function () {
  const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
  if (tg) { tg.expand(); tg.ready(); }

  const urlParams = new URLSearchParams(window.location.search);
  let roomId = urlParams.get("room");
  if (!roomId) {
    roomId = "TEST" + Math.random().toString(36).slice(2, 6).toUpperCase();
  }

  let userId, userName;
  if (tg && tg.initDataUnsafe && tg.initDataUnsafe.user) {
    userId = tg.initDataUnsafe.user.id;
    userName = tg.initDataUnsafe.user.first_name || "بازیکن";
  } else {
    userId = Number(localStorage.getItem("hokm_guest_id") || Math.floor(Math.random() * 1e9));
    localStorage.setItem("hokm_guest_id", userId);
    userName = "مهمان" + String(userId).slice(-3);
  }

  const SUIT_SYMBOLS = { S: "♠", H: "♥", D: "♦", C: "♣" };
  const RED_SUITS = new Set(["H", "D"]);

  const lobbyEl = document.getElementById("lobby");
  const lobbyStatusEl = document.getElementById("lobby-status");
  const seatListEl = document.getElementById("seat-list");
  const tableEl = document.getElementById("table");
  const handEl = document.getElementById("hand");
  const logEl = document.getElementById("log");
  const scoreAEl = document.getElementById("score-a");
  const scoreBEl = document.getElementById("score-b");
  const hokmIndicatorEl = document.getElementById("hokm-indicator");
  const turnBannerEl = document.getElementById("turn-banner");
  const hokmModal = document.getElementById("hokm-modal");
  const roundModal = document.getElementById("round-modal");
  const roundModalText = document.getElementById("round-modal-text");
  const gameoverModal = document.getElementById("gameover-modal");
  const gameoverText = document.getElementById("gameover-text");

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/${roomId}`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ action: "join", user_id: userId, name: userName }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "error") {
      lobbyStatusEl.textContent = "❌ " + msg.message;
      return;
    }
    if (msg.type === "state") {
      render(msg.state);
    }
  };

  ws.onclose = () => {
    lobbyStatusEl.textContent = "اتصال قطع شد. صفحه رو رفرش کن.";
  };

  function cardEl(card) {
    const div = document.createElement("div");
    div.className = "card " + (RED_SUITS.has(card.slice(-1)) ? "red" : "black");
    const rank = card.slice(0, -1);
    const suit = card.slice(-1);
    div.innerHTML = `<span class="rank">${rank}</span><span class="suit">${SUIT_SYMBOLS[suit]}</span>`;
    return div;
  }

  function render(state) {
    if (state.your_seat === null || state.your_seat === undefined) {
      lobbyStatusEl.textContent = "در حال ورود به میز...";
      return;
    }

    const seatedCount = state.seats.filter(Boolean).length;
    if (state.phase === "waiting") {
      lobbyEl.classList.remove("hidden");
      tableEl.classList.add("hidden");
      lobbyStatusEl.textContent = `منتظرِ بازیکنا... (${seatedCount}/۴) — کدِ میز: ${state.room_id}`;
      seatListEl.innerHTML = "";
      state.seats.forEach((s, i) => {
        const chip = document.createElement("div");
        chip.className = "seat-chip" + (s ? " filled" : "");
        chip.textContent = s ? s.name : `صندلی ${i + 1} خالی`;
        seatListEl.appendChild(chip);
      });
      return;
    }

    lobbyEl.classList.add("hidden");
    tableEl.classList.remove("hidden");

    const yourSeat = state.your_seat;

    for (let actual = 0; actual < 4; actual++) {
      const rel = (actual - yourSeat + 4) % 4;
      const seatDiv = document.getElementById("seat-" + rel);
      const info = state.seats[actual];
      const nameDiv = seatDiv.querySelector(".seat-name");
      const countDiv = seatDiv.querySelector(".card-count");
      nameDiv.textContent = info ? info.name + (info.connected ? "" : " (قطع)") : "—";
      countDiv.textContent = info ? `${info.card_count} کارت` : "";
      seatDiv.classList.toggle("active", state.turn_seat === actual && state.phase === "playing");

      const trickDiv = document.getElementById("trick-" + rel);
      trickDiv.innerHTML = "";
      const cardOnTable = state.current_trick[actual];
      if (cardOnTable) trickDiv.appendChild(cardEl(cardOnTable));
    }

    scoreAEl.textContent = state.round_points[0];
    scoreBEl.textContent = state.round_points[1];
    hokmIndicatorEl.textContent = state.hokm_suit ? "حکم: " + SUIT_SYMBOLS[state.hokm_suit] : "حکم: —";

    if (state.phase === "playing") {
      const turnRel = (state.turn_seat - yourSeat + 4) % 4;
      turnBannerEl.textContent = turnRel === 0 ? "نوبتِ توئه!" : `نوبتِ ${state.seats[state.turn_seat]?.name || "..."}`;
    } else {
      turnBannerEl.textContent = "";
    }

    logEl.textContent = (state.log || []).slice(-2).join(" | ");

    handEl.innerHTML = "";
    const legalSet = new Set(state.legal_cards || []);
    const canPlay = state.phase === "playing" && state.turn_seat === yourSeat;
    (state.your_hand || []).forEach((card) => {
      const el = cardEl(card);
      if (canPlay) {
        el.classList.add(legalSet.has(card) ? "legal" : "illegal");
        if (legalSet.has(card)) {
          el.addEventListener("click", () => {
            ws.send(JSON.stringify({ action: "play_card", card }));
          });
        }
      }
      handEl.appendChild(el);
    });

    if (state.phase === "choosing_hokm" && state.hakem_seat === yourSeat) {
      hokmModal.classList.remove("hidden");
    } else {
      hokmModal.classList.add("hidden");
    }

    if (state.phase === "round_over") {
      roundModal.classList.remove("hidden");
      roundModalText.textContent =
        `دور تموم شد! امتیازها: تیمِ A ${state.round_points[0]} — تیمِ B ${state.round_points[1]}`;
    } else {
      roundModal.classList.add("hidden");
    }

    if (state.phase === "game_over") {
      gameoverModal.classList.remove("hidden");
      const winner = state.round_points[0] > state.round_points[1] ? "A" : "B";
      gameoverText.textContent = `🏆 تیمِ ${winner} بردِ کامل رو گرفت! (${state.round_points[0]} - ${state.round_points[1]})`;
    } else {
      gameoverModal.classList.add("hidden");
    }
  }

  document.querySelectorAll(".suit-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      ws.send(JSON.stringify({ action: "choose_hokm", suit: btn.dataset.suit }));
    });
  });

  document.getElementById("next-round-btn").addEventListener("click", () => {
    ws.send(JSON.stringify({ action: "next_round" }));
  });
})();
