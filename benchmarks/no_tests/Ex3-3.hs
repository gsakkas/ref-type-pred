module Ex3_3 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ type TRUE = {v:Bool | v } @-}
{-@ type FALSE = {v:Bool | not v} @-}

{-@ lAssert :: <mask> @-}
lAssert :: Bool -> a -> a
lAssert True x = x
lAssert False _ = die "yikes, assertion fails!"