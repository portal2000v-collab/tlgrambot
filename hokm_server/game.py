"""
منطق کاملِ بازیِ حکمِ چهارنفره (بدونِ رابط کاربری). هر «اتاق» یه نمونه از HokmGame داره.
"""

import random
from enum import Enum


SUITS = ["S", "H", "D", "C"]  # ♠ ♥ ♦ ♣
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}

TRICKS_PER_ROUND = 13
POINTS_TO_WIN = 7


class Phase(str, Enum):
    WAITING = "waiting"
    CHOOSING_HOKM = "choosing_hokm"
    PLAYING = "playing"
    ROUND_OVER = "round_over"
    GAME_OVER = "game_over"


def make_deck():
    deck = [f"{r}{s}" for s in SUITS for r in RANKS]
    random.shuffle(deck)
    return deck


def card_rank(card: str) -> str:
    return card[:-1]


def card_suit(card: str) -> str:
    return card[-1]


def sort_hand(hand):
    return sorted(hand, key=lambda c: (card_suit(c), RANK_VALUE[card_rank(c)]))


class HokmGame:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.seats = [None, None, None, None]
        self.hands = [[], [], [], []]
        self.hakem_seat = None
        self.hokm_suit = None
        self.turn_seat = None
        self.current_trick = [None, None, None, None]
        self.lead_suit = None
        self.tricks_won = [0, 0]
        self.round_points = [0, 0]
        self.phase = Phase.WAITING
        self.log = []
        self.first_hakem_chosen = False

    def seat_of(self, user_id):
        for i, s in enumerate(self.seats):
            if s and s["user_id"] == user_id:
                return i
        return None

    def join(self, user_id, name):
        existing = self.seat_of(user_id)
        if existing is not None:
            self.seats[existing]["name"] = name
            return existing
        for i in range(4):
            if self.seats[i] is None:
                self.seats[i] = {"user_id": user_id, "name": name, "connected": True}
                return i
        return None

    def mark_disconnected(self, user_id):
        i = self.seat_of(user_id)
        if i is not None:
            self.seats[i]["connected"] = False

    def all_seated(self):
        return all(s is not None for s in self.seats)

    def team_of(self, seat):
        return seat % 2

    def start_round(self):
        deck = make_deck()
        if self.hakem_seat is None:
            self.hakem_seat = random.randint(0, 3)
        self.hands = [[], [], [], []]
        for i in range(4):
            self.hands[i] = sort_hand(deck[i * 13:(i + 1) * 13])
        self.hokm_suit = None
        self.tricks_won = [0, 0]
        self.current_trick = [None, None, None, None]
        self.lead_suit = None
        self.turn_seat = self.hakem_seat
        self.phase = Phase.CHOOSING_HOKM
        self.log.append(f"دورِ جدید شروع شد. حاکم: {self.seats[self.hakem_seat]['name']}")

    def choose_hokm(self, seat, suit):
        if self.phase != Phase.CHOOSING_HOKM or seat != self.hakem_seat or suit not in SUITS:
            return False
        self.hokm_suit = suit
        self.phase = Phase.PLAYING
        self.turn_seat = self.hakem_seat
        self.log.append(f"حکم شد: {SUIT_SYMBOLS[suit]}")
        return True

    def legal_cards(self, seat):
        hand = self.hands[seat]
        if self.lead_suit is None:
            return hand
        same_suit = [c for c in hand if card_suit(c) == self.lead_suit]
        return same_suit if same_suit else hand

    def play_card(self, seat, card):
        if self.phase != Phase.PLAYING or seat != self.turn_seat:
            return False
        if card not in self.hands[seat]:
            return False
        if card not in self.legal_cards(seat):
            return False

        self.hands[seat].remove(card)
        self.current_trick[seat] = card
        if self.lead_suit is None:
            self.lead_suit = card_suit(card)

        next_seat = (seat + 1) % 4
        if all(c is not None for c in self.current_trick):
            self._resolve_trick()
        else:
            self.turn_seat = next_seat
        return True

    def _resolve_trick(self):
        winner_seat = None
        best_value = -1
        for i, card in enumerate(self.current_trick):
            suit = card_suit(card)
            value = RANK_VALUE[card_rank(card)]
            if suit == self.hokm_suit:
                value += 100
            if suit != self.hokm_suit and suit != self.lead_suit:
                continue
            if value > best_value:
                best_value = value
                winner_seat = i

        team = self.team_of(winner_seat)
        self.tricks_won[team] += 1
        self.log.append(
            f"{self.seats[winner_seat]['name']} این دست رو با {self.current_trick[winner_seat]} برد."
        )
        self.current_trick = [None, None, None, None]
        self.lead_suit = None
        self.turn_seat = winner_seat

        total_tricks = self.tricks_won[0] + self.tricks_won[1]
        if self.tricks_won[team] >= 7 or total_tricks >= TRICKS_PER_ROUND:
            self._finish_round()

    def _finish_round(self):
        winner_team = 0 if self.tricks_won[0] > self.tricks_won[1] else 1
        loser_team = 1 - winner_team
        kot = self.tricks_won[loser_team] == 0
        self.round_points[winner_team] += 2 if kot else 1
        self.log.append(
            f"دور تموم شد! تیمِ {'A' if winner_team == 0 else 'B'} برد "
            f"({self.tricks_won[winner_team]} به {self.tricks_won[loser_team]})"
            + (" — کُت! ۲ امتیاز." if kot else "")
        )
        loser_seats = [s for s in range(4) if self.team_of(s) == loser_team]
        self.hakem_seat = random.choice(loser_seats)

        if self.round_points[winner_team] >= POINTS_TO_WIN:
            self.phase = Phase.GAME_OVER
            self.log.append(f"تیمِ {'A' if winner_team == 0 else 'B'} برنده‌ی کلِ بازی شد! 🏆")
        else:
            self.phase = Phase.ROUND_OVER

    def continue_next_round(self):
        if self.phase == Phase.ROUND_OVER:
            self.start_round()

    def bot_take_turn_if_needed(self):
        if self.phase == Phase.CHOOSING_HOKM:
            seat = self.hakem_seat
            if self.seats[seat] and not self.seats[seat].get("connected", True):
                self.choose_hokm(seat, random.choice(SUITS))
            return
        if self.phase == Phase.PLAYING:
            seat = self.turn_seat
            if self.seats[seat] and not self.seats[seat].get("connected", True):
                card = random.choice(self.legal_cards(seat))
                self.play_card(seat, card)

    def public_state(self, viewer_user_id=None):
        viewer_seat = self.seat_of(viewer_user_id) if viewer_user_id is not None else None
        return {
            "room_id": self.room_id,
            "phase": self.phase.value,
            "seats": [
                None if s is None else {
                    "name": s["name"], "connected": s.get("connected", True),
                    "card_count": len(self.hands[i]),
                }
                for i, s in enumerate(self.seats)
            ],
            "your_seat": viewer_seat,
            "your_hand": sort_hand(self.hands[viewer_seat]) if viewer_seat is not None else [],
            "hakem_seat": self.hakem_seat,
            "hokm_suit": self.hokm_suit,
            "turn_seat": self.turn_seat,
            "current_trick": self.current_trick,
            "tricks_won": self.tricks_won,
            "round_points": self.round_points,
            "legal_cards": (
                self.legal_cards(viewer_seat)
                if viewer_seat is not None and self.phase == Phase.PLAYING and viewer_seat == self.turn_seat
                else []
            ),
            "log": self.log[-12:],
        }
