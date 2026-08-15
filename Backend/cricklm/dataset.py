"""
dataset.py — CrickLM Dataset

"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import CrickTokenizer

SYNTHETIC_CORPUS = """
cricket is a bat and ball sport played between two teams of eleven players on a field.
the batting team scores runs by striking the ball bowled at the wicket.
the bowler delivers the ball from one end of the pitch toward the batsman.
the batsman attempts to hit the ball and score runs by running between the wickets.

virat kohli plays a magnificent cover drive down the ground. the ball races to the boundary.
rohit sharma pulls the short ball over deep square leg for a massive six.
the bowler sends down a full length delivery on the off stump line.
the batsman comes forward and drives the ball through the covers.
that is a beautiful straight drive off a good length delivery.
the spin bowler flighted the ball beautifully and the batsman was deceived in the flight.
the batsman stepped down the track and lofted the ball over long on.
excellent yorker from the fast bowler right at the base of the stumps.
the batsman digs it out with a superb last minute adjustment.
full toss outside the off stump and the batsman cuts it to the boundary.
the batsman plays the late cut and finds the gap between point and gully.
short pitched delivery and the batsman hooks it over fine leg.
the seam bowler swings it late and beats the outside edge of the bat.
no ball called by the umpire as the bowler overstepped the crease.

biomechanical analysis shows the batsman has excellent head position.
the front foot is planted correctly at the pitch of the ball.
the bat swing follows a straight path creating a solid base for contact.
elbow position during the cover drive should be high for better control.
the back foot drive requires weight transfer to the back foot early.
head stability throughout the shot determines accuracy of ball striking.
knee bend depth influences the power transfer from lower body to bat.
the follow through of the bat reveals the intended direction of the shot.
shoulder alignment at the point of delivery is crucial for pace bowlers.
the bowling arm should be high at the point of release for maximum bounce.
front knee should be braced at the point of release for an effective action.
the run up must be rhythmic and consistent to build momentum before delivery.
wrist position at release determines the amount of seam movement.
the non bowling arm guides the body to stay sideways during delivery.
hip rotation generates significant power in the bowling action.

tactical analysis reveals the batsman has a weakness outside the off stump.
he tends to play away from his body when the ball is pitched on a length.
the opposition bowling attack should target the corridor of uncertainty.
a short pitched delivery into the ribs would be effective against this batsman.
the reverse swing bowler will be dangerous in these dry conditions.
field placing shows three slips and a gully for the opening bowlers.
the captain has set an attacking field with men catching positions.
defensive field placing with sweepers on the boundary protects the target.
the spin bowler has a man at short extra cover to cut off the drive.
deep square leg is in position anticipating the pull shot attempt.
the bowling coach recommends a fuller length to restrict scoring opportunities.
the batting coach advises the player to play closer to the body.
watch the ball right onto the bat is the most fundamental coaching advice.
get your foot to the pitch of the ball before committing to the drive.

pace bowling requires building pressure through consistent line and length.
the inswing bowler targets the pads to force the batsman onto the back foot.
off spin bowling uses drift and dip to deceive the batsman in the air.
leg spin is the most attacking form of spin bowling in modern cricket.
the googly deceives the batsman by turning the opposite way to the leg break.
a flipper is bowled from the front of the hand and skids through quickly.
the doosra turns away from the right handed batsman like an off break.
a carrom ball is flicked from the middle finger and can turn either way.
swing bowling relies on keeping one side of the ball shiny and polished.
reverse swing occurs when the ball is old and the rough side faces forward.
the seam position determines the direction of conventional swing.
a cutterseam bowler uses the fingers to move the ball off the pitch.
the bouncer is a short pitched delivery aimed at the batsman upper body.
the slower ball deceives the batsman who has committed to an attacking shot.
a knuckleball is gripped with the knuckles to reduce pace through the air.

the batsman requires front foot balance drills to improve footwork.
shadow batting practice helps the player rehearse shot execution without a ball.
throw down sessions build confidence against short pitched deliveries.
throw downs off a tee improve contact quality and bat swing consistency.
reaction ball training sharpens reflexes against unpredictable movement.
video analysis helps players identify technical flaws in their technique.
strength and conditioning training builds the core stability needed for batting.
rotational strength in the hips generates power through the ball striking zone.
grip strength exercises improve bat control in difficult conditions.
yoga and flexibility work prevents injury and maintains range of motion.
mental conditioning helps players handle pressure situations in matches.
visualization techniques allow players to rehearse shots in the mind.
breathing exercises control heart rate under pressure during key moments.

first innings total of two hundred and eighty five gives the team a competitive score.
the pitch is showing some signs of wear and spin is expected later in the day.
overcast conditions will assist the swing bowlers in the morning session.
the toss was won by the home side who chose to bat first on this surface.
powerplay overs are crucial for setting up a solid platform in limited overs cricket.
the death overs specialist bowler conceded only six runs from the final over.
two wickets fell in quick succession to put the batting team under pressure.
partnership building is essential to recover from the early wicket losses.
running between the wickets can add valuable runs through quick singles.
rotating the strike keeps the scoreboard moving and maintains momentum.
the new ball was taken immediately after the mandatory thirty over break.
the umpire referred a caught behind appeal to the third umpire for review.
ball tracking technology showed the ball would have hit the top of off stump.
the soft signal from the on field umpire was out before the review check.
decision review system has changed the way the game is played and umpired.

test match cricket demands the highest technical standards from all players.
the opening batsman must see off the new ball in difficult early conditions.
graft and determination are as important as natural talent at the highest level.
the spinner must be patient and wait for the batsman to make an error.
building pressure over long spells requires exceptional physical fitness.
the fielding team set an attacking ring to cut off the easy singles.
the captain brought himself on to bowl after sensing an opportunity.
excellent fielding and catching reflect the team's preparation and standards.
a run out can change the course of a match with momentum shifting suddenly.
direct hit from the deep midwicket fielder finds the batsman well short.
stumped down the leg side after the batsman missed a wide spinning delivery.
caught at slip after the ball moved late off the seam to take the outside edge.
lbw decision given after the ball pitched in line and struck the front pad.
bowled through the gate as the ball nipped back between bat and pad.
hit wicket as the batsman disturbed the stumps while playing the pull shot.

the cricket academy focuses on developing young talent through structured programs.
junior cricketers learn the fundamentals of batting bowling and fielding systematically.
the coaching staff uses technology and data to track player development over time.
biomechanical modelling identifies technical flaws and suggests corrective exercises.
sports science integrates physical mental and technical preparation for peak performance.
nutrition planning supports the energy demands of professional cricket training.
recovery protocols after intense training sessions maintain player availability.
sleep quality directly impacts cognitive performance and reaction time in match play.
hydration management is critical especially in hot subcontinental match conditions.
the physio works closely with the coaching staff to manage player workloads carefully.
"""


def build_corpus(extra_files: Optional[List[str]] = None) -> str:
    corpus_path = Path("data/cricket_corpus.txt")
    parts = []

    
    if corpus_path.exists():
        text = corpus_path.read_text(encoding="utf-8")
        parts.append(text)
        print(f"Loaded corpus: {corpus_path} ({len(text):,} chars)")

    if extra_files:
        for fp in extra_files:
            p = Path(fp)
            if p.exists():
                text = p.read_text(encoding="utf-8")
                parts.append(text)
                print(f"Loaded extra: {fp} ({len(text):,} chars)")

    parts.append(SYNTHETIC_CORPUS)
    print(f"Included synthetic corpus ({len(SYNTHETIC_CORPUS):,} chars)")

    full = "\n\n".join(parts)
    print(f"Total corpus size: {len(full):,} characters")
    return full


# PyTorch Dataset
class CricketDataset(Dataset):
    def __init__(
        self,
        tokenizer: CrickTokenizer,
        corpus: str,
        seq_len: int = 128,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        print(f"Tokenizing corpus for dataset (seq_len={seq_len})...")
        self.token_ids = tokenizer.encode(corpus, add_bos=False, add_eos=False)
        print(f"  Total tokens: {len(self.token_ids):,}")
        print(f"  Total samples: {len(self):,}")

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input token IDs  shape (seq_len,)
            y: Target token IDs shape (seq_len,)  — x shifted right by 1
        """
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)   
        y = torch.tensor(chunk[1:],  dtype=torch.long)   
        return x, y


def make_dataloader(
    tokenizer: CrickTokenizer,
    corpus: str,
    seq_len: int = 128,
    batch_size: int = 32,
    shuffle: bool = True,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    full_dataset = CricketDataset(tokenizer, corpus, seq_len)

    n = len(full_dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    
    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}")
    return train_loader, val_loader