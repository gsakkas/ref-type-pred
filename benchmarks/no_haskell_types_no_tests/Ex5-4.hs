module Ex5_4 where

{-@ data IncList a = Emp
                   | (:<) { hd :: a, tl :: IncList {v:a | hd <= v}} @-}
data IncList a = Emp
               | (:<) { hd :: a, tl :: IncList a }
               deriving (Eq)

infixr 9 :<

{-@ append :: <mask> @-}
append z Emp ys = z :< ys
append z (x :< xs) ys = x :< append z xs ys