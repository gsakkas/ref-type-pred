module Ex9_1_0 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ data SList a = SL { size :: Nat, elems :: {v:[a] | realSize v = size}} @-}
data SList a = SL { size :: Int, elems :: [a] } deriving (Show, Eq)

{-@ measure realSize @-}
realSize :: [a] -> Int
realSize [] = 0
realSize (_:xs) = 1 + realSize xs

{-@ type SListN a N = {v:SList a | size v = N} @-}

{-@ tl :: xs:{SList a | size xs > 0} -> SListN a {size xs - 1} @-}
tl (SL n (_:xs)) = SL (n-1) xs
tl _ = die "empty SList"

{-@ test1 :: SListN String 0 @-}
test1 = tl (SL 1 ["cat"])
test1' = tl (SL 1 ["cat"]) == SL 0 []

{-@ test2 :: SListN String 1 @-}
test2 = tl (SL 2 ["cat", "dog"])
test2' = tl (SL 2 ["cat", "dog"]) == SL 1 ["dog"]

-- {-@ test3 :: {v:_ | false} @-}
-- test3 = tl (SL 0 [])
-- test3' = tl (SL 0 []) == undefined