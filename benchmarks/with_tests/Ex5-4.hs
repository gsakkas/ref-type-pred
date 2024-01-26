module Ex5_4 where

{-@ data IncList a = Emp
                   | (:<) { hd :: a, tl :: IncList {v:a | hd <= v}} @-}
data IncList a = Emp
               | (:<) { hd :: a, tl :: IncList a }
               deriving (Eq)

infixr 9 :<

{-@ append :: <mask> @-}
append :: (Ord a) => a -> IncList a -> IncList a -> IncList a
append z Emp ys = z :< ys
append z (x :< xs) ys = x :< append z xs ys

-- {-@ test1 :: {v:_ | false} @-}
-- test1 :: IncList Int
-- test1 = append 5 (1 :< 2 :< Emp) (4 :< Emp)
-- test1' = test1 == undefined

{-@ test2 :: IncList Int @-}
test2 :: IncList Int
test2 = append 3 (1 :< 2 :< Emp) (4 :< Emp)
test2' = test2 == (1 :< 2 :< 3 :< 4 :< Emp)