# Run report: `rnc1_tinystories_shakedown_001`

Config: 12L d768 h12 mlp2048 vocab 8192 ctx 512; precision bf16, attention auto, lr 0.0006, warmup 150, total_steps 3000, tokens/step 32,768.

Parameters: 91,246,848. GPU: AMD Radeon RX 9060 XT (gfx1200), torch 2.9.1+rocmsdk20260116. Tokenizer sha256 `d37360ce034918fc…`. Corpus sha256 `c5cf5e22ff13614e…`.

Status: **completed** at step 3000, 98,304,000 tokens seen, last segment elapsed 4973.1 s.

## Held-out evaluation over time

| step | tokens seen | val loss (nats) | val ppl | bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 9.1523 | 9435.8 | 3.137 |
| 250 | 8,192,000 | 2.5392 | 12.7 | 0.870 |
| 500 | 16,384,000 | 2.1146 | 8.3 | 0.725 |
| 750 | 24,576,000 | 1.9439 | 7.0 | 0.666 |
| 1000 | 32,768,000 | 1.8394 | 6.3 | 0.630 |
| 1250 | 40,960,000 | 1.7631 | 5.8 | 0.604 |
| 1500 | 49,152,000 | 1.7029 | 5.5 | 0.584 |
| 1750 | 57,344,000 | 1.6545 | 5.2 | 0.567 |
| 2000 | 65,536,000 | 1.6145 | 5.0 | 0.553 |
| 2250 | 73,728,000 | 1.5787 | 4.8 | 0.541 |
| 2500 | 81,920,000 | 1.5531 | 4.7 | 0.532 |
| 2750 | 90,112,000 | 1.5338 | 4.6 | 0.526 |
| 3000 | 98,304,000 | 1.5221 | 4.6 | 0.522 |

## Training loss (every logged step, thinned)

| step | loss | lr | grad norm | tok/s | peak mem GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.1584 | 4.00e-06 | 10.588 | 11396 | 6.83 |
| 120 | 3.2687 | 4.80e-04 | 1.205 | 20512 | 7.52 |
| 240 | 2.5119 | 5.99e-04 | 0.529 | 20809 | 7.52 |
| 360 | 2.2626 | 5.93e-04 | 0.451 | 20330 | 7.52 |
| 480 | 2.1060 | 5.82e-04 | 0.418 | 20124 | 7.52 |
| 600 | 2.0601 | 5.68e-04 | 0.384 | 20341 | 7.52 |
| 720 | 1.9619 | 5.49e-04 | 0.345 | 20474 | 7.52 |
| 840 | 1.9864 | 5.26e-04 | 0.334 | 20469 | 7.52 |
| 960 | 1.9213 | 5.00e-04 | 0.327 | 20401 | 7.52 |
| 1080 | 1.7547 | 4.70e-04 | 0.314 | 20217 | 7.52 |
| 1200 | 1.7122 | 4.39e-04 | 0.315 | 20487 | 7.52 |
| 1320 | 1.7834 | 4.05e-04 | 0.328 | 20450 | 7.52 |
| 1440 | 1.7015 | 3.70e-04 | 0.307 | 20765 | 7.52 |
| 1560 | 1.6967 | 3.35e-04 | 0.304 | 20258 | 7.52 |
| 1680 | 1.6767 | 2.99e-04 | 0.309 | 20376 | 7.52 |
| 1800 | 1.6073 | 2.64e-04 | 0.304 | 20332 | 7.52 |
| 1920 | 1.5166 | 2.30e-04 | 0.299 | 20475 | 7.52 |
| 2040 | 1.5799 | 1.98e-04 | 0.294 | 20294 | 7.52 |
| 2160 | 1.5727 | 1.68e-04 | 0.301 | 20763 | 7.52 |
| 2280 | 1.5475 | 1.41e-04 | 0.295 | 20799 | 7.52 |
| 2400 | 1.4732 | 1.17e-04 | 0.296 | 20456 | 7.52 |
| 2520 | 1.5624 | 9.71e-05 | 0.308 | 20755 | 7.52 |
| 2640 | 1.5585 | 8.11e-05 | 0.302 | 20728 | 7.52 |
| 2760 | 1.5878 | 6.95e-05 | 0.308 | 20773 | 7.52 |
| 2880 | 1.5002 | 6.24e-05 | 0.307 | 20598 | 7.52 |
| 3000 | 1.4305 | 6.00e-05 | 0.306 | 20894 | 7.52 |

## Throughput and memory

- tokens/s: median 20470, min 19182, max 21835
- seconds/step: median 1.595
- peak GPU memory allocated: 7.52 GB; reserved: 7.96 GB
- wall time logged: 82.7 min (this segment)
- grad norm: first 10.588, median 0.312, max 10.588; steps with grad norm > clip (1.0): 11/301 logged

## Samples at fixed prompts (temperature 0.8, top-k 50, fixed noise seed)

### step 0

- **Once upon a time, there was a little** knock weird outfit throatperate smooth smooth shot three reapp throat throathhstorm changing shopping shoppingamie squeezed ban breaths A A A eventuallyarth� envelope envelope envelope wellusie well beggedilarth Kelly gasped Suddenly squeezedarliearthgetable poke throat arrowshh gazmejam Max crumbs envelope envelopeessSome supphh alither envelope envelope delayedigning envelope prett yawned arms yawned Johnson farmcer pajamashic barked outfit merma prints veterinarian circles learned conversation crumbs melting exha three distractperate kneel anywayoth well well barkedale shocked anywhere�esscle'dill did gazed pawhew'd drain exha exha purposeJessoot barked exha throat Johnson arms disappoininked  _(120 tokens)_
- **The dog was very** tough playtime playtime vi interested chimneyoud dolphin afford Nemo climbedue sofa dri dri drio complainolol Rock Rock meow knight passport searched eventually apisten snowball expected� snowballeel� every passport8 circles tested silentoraoud cautiouslyoral circles explains rhythm human gloomy complainmble8 apologized present dolphin complainark expected expected circles regular regular oil mysterious tightly oil oil oil doggy meow stri wiggled silent Buddy embarrass embarrass called8 twinklingue vi circles circlesesmer Alongience introduced harb burst valuedimp introducedimp wisely present embarrass become cact touruffy burst dolphin bowls bring hills repa love slowed slowed envelope tour crumbs embarrass embarrassoud presenteeling hawk8  _(120 tokens)_
- **One day, Lily went to the park and** believing anywhere suggestion bask bask math squeezed losethhnny]]]] shopping homework scraped shopping shopping taxi takingling Kim painterosed sprinkles suitcase show'd television yours forg bicyclefuowersowers."elery drops fire fire responsible rubbing"D badge suitcase shopping shopping shopping knot shopping trains shopping"Ben� hangilla muffins muffins crumbsfu"DTheirippers badgeJake presents called Ellie forest sneak sneak fr lose hardlyFoMar!” noodles noodlesHer encouraged fr salad counterL daring homework jungle squiraveippersippers remem taking homeworkippersTheir twist."." ribbons sneeze deal." where sneak fr lives scooters shopping Sadie gone road pins prett shopping shopping smooth competition  _(120 tokens)_
- **Tom and his mom** mom focove open mor’m’m embarrass] dancing rubbingining"Okay rubbed fauc coins claps claps marked open seashell paints snowball snowballoud shap gazed smoothlcanooudoud� MaryMom noticedsc deliverhelloienienag dolphin dolphin dolphinletbulbulem snugg snugg saMom bicycleTo crumbs paints sa brushesbul seasbulletexuly"Surek signed scrubbedination homework coins salad scrubbed coins coins snowball own rubbed eventually rubbing accidentallyins understand storekeeper scootersoom rubbing painterFredFredhello fauc gonna action This tw called] Princessurpr medicine closer smooth zigzag dolphin ring"Sure board board fog marks leaninghello fauc collecting Laura guest crumbs seas sausage  _(120 tokens)_

### step 250

- **Once upon a time, there was a little** girl named Lucy. She was three years old and loved to watch the world. One day, she went to the market with her parents.  ⏎ At the museum, they saw a big, old man walking in the park. Lucy was scared. But then, she remembered the old man had a long stick. Lucy was so happy to see him. She didn't want to stop until she finally got back to the park.  ⏎ Lucy was determined to go outside and ran back to her mom. Her mom tried to hold the man's hand, but it was too late. Lucy was so hurt,  _(120 tokens)_
- **The dog was very** happy. "What do you do? Go to the park with me?" he said. "I'll give you a new ball. I can throw the ball around you." ⏎ The dog tried to help the dog, but it was too high. It was getting stuck! The dog could not reach the carrot. It was a big ball of dirt. The dog tried to push the ball, but he couldn't. The dog cried and they ran away. ⏎ The dog and the dog were never to give up again. They had a lot of fun. They did not know if they had seen the dog  _(120 tokens)_
- **One day, Lily went to the park and** saw a big slide. She wanted to climb up in the grass, but her mommy said it was too late. They got on the swing and Lily was sad. She cried and screamed. Her mommy was very sad and angry. ⏎ Lily didn't know what to do. She felt sorry for her mommy and daddy. She looked in the park with her teddy bear. She saw that Lily was upset and hurt. She wished she had not played with her friends. She wished he had never touched her arm. She wished she had listened to her mommy and to be happy.  _(115 tokens, ended with <|eos|>)_
- **Tom and his mom** went to the kitchen. They looked at the stove. It was dark. ⏎ "Look, there is a big pot!" Tom said. "I made it!" ⏎ "OK, Mom, I have this pie. It is my favorite soup. It is sweet and tasty. It has more fruit for us." ⏎ But they did not see a cup. It was brown and shiny. It was not dirty. It made a loud noise. It was black and soft. ⏎ "Let me have a cake," Tom said. He wanted to eat a yummy cake. He took a slice and milk and sugar. He walked to the  _(120 tokens)_

### step 500

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore. One day, she decided to go for a walk in the park. She was wearing her favorite blue shoes and the feeling of fun. ⏎ When she came to a big tree, she found something amazing. It was a bright, long pink ball. Lucy was so happy and wanted to have it. She decided to swing it on her own, and it was so much fun. ⏎ Lucy started to slide down the tree, throwing it until it was so heavy! The girl was so surprised and happy. She could not keep up,  _(120 tokens)_
- **The dog was very** happy. "Wow, look at my dog!" he shouted. He licked his nose and said, "I have a new collar. It's so cool!" ⏎ The dog wagged his tail and licked his owner's face. The dog's owner was so happy. He hugged the dog and thanked him. "Thank you for the day!" his owner said. The dog wagged his tail and licked his owner's face.  _(84 tokens, ended with <|eos|>)_
- **One day, Lily went to the park and** saw a big tree. She wanted to climb it, but it was difficult. So, she decided to climb the tree and see what was happening. ⏎ As Lily sat down, a bird flew by. She peeked up and saw a small bird and asked, "Can you help me climb the tree?" The bird replied, "Sure, I will help you!" ⏎ Lily climbed down the tree and started to climb. The bird followed her and showed her how to climb the tree. She was so happy and thanked the bird for helping her. From that day on, Lily learned that even though she might have  _(120 tokens)_
- **Tom and his mom** both wanted the same. They looked at each other and frowned. He wanted to see what was happening and his mom said. ⏎ "It's okay, but you know why you both wanted the big blue ball. Let's go and find it," his mom said. ⏎ Tom nodded and agreed. He put the ball in his mouth and tried to hit the ball. He felt his lips and his tail hurt. He was curious and excited. ⏎ "Tom, I'm sorry, mom. I didn't want to break my ball," he said. ⏎ His mom smiled and hugged him. She said, "It's  _(120 tokens)_

### step 750

- **Once upon a time, there was a little** girl named Lucy. She loved to play outside in the sun every day. One day, she saw a big, dark cloud in the sky. She knew she had to protect it from the dark sky. ⏎ Lucy went inside her house and found a cozy spot under a tree. She had so much fun playing in the rain. She knew that her mommy and daddy were right. She played in the rain all day, and that was the fun she had with her family when it was time to go back inside. ⏎ Lucy was happy that she could help her family and have fun with her family again. This  _(120 tokens)_
- **The dog was very** happy. "What do you need to do?" the man asked. "I need a lot of things for my friends," the dog replied. He found a tree and started to build his house. He worked hard, but he was too tired. The dog was feeling sad. He wanted his house and his friends and family. He wanted his friends back. ⏎ Then, he had an idea. He went to his friend and asked what they needed. His friend said, "We can make your house look more beautiful." The dog agreed and he was happy. His home was even more fancy now.  _(120 tokens)_
- **One day, Lily went to the park and** saw a big tree. She wanted to climb the tree, but she was scared. She asked her mom, "Can we go up the tree, please?" ⏎ Her mom said, "No, Lily, we can't. We have to get down and leave our house. We have to wait for our turn." ⏎ Lily was sad. She did not understand why her mom said that. She wanted to make her mom happy. She asked her mom, "But mom, can we go down the tree? We can play?" ⏎ Her mom said, "No, Lily, we have to wait for the  _(120 tokens)_
- **Tom and his mom** went to the park. They saw a big tree with lots of grass. They wanted to see what was under the tree. ⏎ Tom and his mom walked to the tree. They looked around and found a worm. He said, "Look, mom, a worm!" Tom was so happy! He ran around the park and saw a bird's nest. The bird looked sad. ⏎ Tom and his mom walked around the park. They saw a big tree with a shiny feather on it. They knew they could use the feather to draw a card. Tom and his mom picked out a long stick. They tied  _(120 tokens)_

### step 1000

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore the woods near her home. ⏎ One day, while she was playing, Lucy heard a noise in the bushes. It was coming from a bush filled with delicious oranges. She was curious and wanted to know what it was. ⏎ Lucy approached the bush and saw that it was full of oranges. "What are you doing here?" she asked herself. ⏎ The bush smiled and said, "I'm gathering acorns when I'm hungry, so I'd never find any to eat". ⏎ Lucy was determined to find some berries, so she gathered some of  _(120 tokens)_
- **The dog was very** happy. "Woof, woof!" he said. "I love to play with you, dog." ⏎ One day, the dog wanted to play. He found a stick and started to throw it near his mom. She saw the dog and said, "Wow, you are so lucky!" ⏎ The dog wagged his tail and barked. "I want to play with you, Lily!" ⏎ Mom smiled and said, "I hope you get good and they will play together forever." ⏎ Lily smiled and said, "Yes, I can!"  _(105 tokens, ended with <|eos|>)_
- **One day, Lily went to the park and** saw a big barrel. She wanted to fill her water with water, so she ran to the barrel. She filled the barrel with water and filled it with water. Now, she was so happy with her water. ⏎ While she was playing, a little bee came by and saw the pot of water. The bee wanted to take it, but Lily was afraid. She decided to use her little beak to make the water clean in her garden. ⏎ Lily walked to the tree and started to clean the water. She washed the wet water and made it look nice. The water was clean and smelled nice. Lily  _(120 tokens)_
- **Tom and his mom** were busy in the living room. Tom saw a big box of crayons on a table. He wanted to draw on them. ⏎ "Look, mom, I found a pretty crayon!" Tom said. He had never seen another crayons before. ⏎ "Mom, can we use that eraser?" Tom asked. "Can we have a crayon, dad?" ⏎ "Sure, Tom. The eraser does not belong here. You have to wait until your eyes and your fingers are very dry. Then you can draw a picture of me and your heart," Dad said. He smiled and took a crayon from Tom's hand. ⏎ Tom took a  _(120 tokens)_

### step 1250

- **Once upon a time, there was a little** girl named Lucy. She was three years old and loved to watch the world. One day, she was feeling very sad and started to cry. ⏎ Her mom asked her, "What's wrong, Lucy?" Lucy replied, "I'm feeling sad, mommy. I want to go to the park. ⏎ Lucy's mom said, "Let's go out and have a picnic!" ⏎ Lucy was excited. She was three years old and so excited. She said yes and they went off to the store. Lucy felt a bit better with her mom watching her. ⏎ At the park, Lucy saw a big tree,  _(120 tokens)_
- **The dog was very** happy. "Can I come to my house now?" he asked. His owner said, "Yes, but remember to stay close so the cat can't move." ⏎ The cat felt happy when he saw the big dog. He wanted to climb up the big tree. So he started to climb the tree. As he was climbing, he saw a big bird. The cat looked sad and scared. The dog wanted to help. ⏎ The cat jumped up and said, "I can't climb down to my nest!" The cat was scared too. But the dog saw the cat was still very helpful. The  _(120 tokens)_
- **One day, Lily went to the park and** saw a big tree. She wanted to climb the tree, but it was too tall for her. She asked her friend, a dog named Max, for help. Max said, "I will help you climb the tree." Lily and Max climbed the tree, and they saw the sky. ⏎ Lily and Max were very happy to be on the tree. They climbed to the top and looked down. "We are so tall!" said Lily. They felt proud and tired, and they had so much fun.  _(102 tokens, ended with <|eos|>)_
- **Tom and his mom** were walking down the street. As they were walking, Tom saw a big dog. It was very loud and it looked very scary.  ⏎ Mom said, "Tom, let's go see the dog." ⏎ They walked slowly past the dog and soon it was right behind them.  ⏎ The dog barked again and started to move around. Tom's mom said, "You will never go near it, Tom. It's scary and it looks scary!" ⏎ But Tom wanted to be brave. He knew that mom was in a hurry. So he got out of the car and continued walking.  ⏎ When they got  _(120 tokens)_

### step 1500

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore. One day, she was walking in her garden when she suddenly saw a big yellow flower. She was so excited, she ran to her mommy and asked, "Mommy, can I pick the yellow flower?"  ⏎ Her mommy looked at her and said, "No, Daisy. That's not right. Go back to your garden and ask my daddy." ⏎ But Lucy wasn't scared. She was a little bit scared, so she stayed behind her mommy. Her daddy started to explain it to her, and explained that the yellow flower  _(120 tokens)_
- **The dog was very** happy. "Woof, woof," he barked. "I love to play with you!" ⏎ One day, a little boy came to see the cat. The cat was very cute and happy. The boy wanted to catch the cat, so he went after the cat. The boy ran after the cat with the cat inside his house. ⏎ But the cat was too fast. The cat ran and ran, trying to catch the boy. The boy was sad and left the cat. The cat was never to be seen again. The end.  _(107 tokens, ended with <|eos|>)_
- **One day, Lily went to the park and** saw a big tree. She wanted to climb it and see what was under the tree. She ran to the tree and climbed on a branch. She felt happy and free. ⏎ But then, she saw a cat walking close to the tree. The cat was black and furry. It liked to play and chase things. Lily was scared of the cat and ran away. ⏎ "Stop! Stop!" she shouted. She chased the cat in her garden. ⏎ "Quick, stop!" Tom called. He chased her around the tree, but he could not reach the cat. ⏎ Lily laughed and ran to her house. She  _(120 tokens)_
- **Tom and his mom** went to the park. Tom liked to run and play. He saw a big hole in the fence. He wanted to see what was inside. He said to his mom, "Look, mom, a big box!" ⏎ His mom was not sure. She said, "Tom, that is not a box. That is a trap. Someone put bad things inside it. It can make you sick. There must be bad bad things inside." ⏎ Tom was scared. He said, "But mom, mom, what is dangerous? What can we do?" ⏎ His mom said, "You can use your hands  _(120 tokens)_

### step 1750

- **Once upon a time, there was a little** girl named Lucy. She loved to play outside in her garden every day. One day, she found a big, thick vine in the ground. She thought it was perfect for climbing up, so she started to climb.  ⏎ As she reached the top of the vine, she saw a big, scary fox. The fox looked hungry and Lucy knew she had to hurry home. She climbed down from the vine and slowly picked up the vine.  ⏎ As Lucy was climbing, she saw a little bird who had a broken wing. The bird could not fly or fly. Lucy felt sad for the bird,  _(120 tokens)_
- **The dog was very** loyal. Every day, the dog would go to the park with his owner. His owner always brought a dog with them.  ⏎ The owner was very compassionate. He was always kind and thoughtful. He made sure the dog got lots of treats before going off to the park.  ⏎ When the dog arrived, he saw lots of other dogs and they were all playing together. He even got a ball of yarn to play tag. The owner was so happy. ⏎ The dog and his owner were having lots of fun in the park. They laughed and played together until the sun started to set.  ⏎ The owner  _(120 tokens)_
- **One day, Lily went to the park and** saw a big slide. She wanted to try it, but it was difficult to climb. She tried and tried, but she couldn't manage. She felt sad. ⏎ A kind boy named Tim saw Lily's sad face and asked, "What's wrong, Lily?" Lily replied, "I don't want to try it, but you can do it." Tim smiled and said, "Don't worry, I can help you!" ⏎ He showed Lily how to climb the big slide. It was hard, but with a little help, Lily climbed up the big slide. Her face turned pale. Lily was  _(120 tokens)_
- **Tom and his mom** were at the store. Tom saw a green dinosaur on the counter. It was very big and had many teeth. Tom asked his mom if he could have it. She said yes, but he had to save money first. ⏎ Tom was very happy. He wanted the green dinosaur. He asked his mom if he could have it. She said yes, but he had to save his money first. Tom was very curious. He asked his mom many questions. She answered all the questions. She said he had to be honest and think about what he wanted. Tom nodded and smiled. He hoped the green  _(120 tokens)_

### step 2000

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore the woods near her home. ⏎ One day, Lucy went to the woods with her mom. They brought a basket with food and water. They spread the blanket on the ground and started to eat. ⏎ Lucy was very hungry so she asked her mom if she could have lunch. Her mom said "Yes, of course!" ⏎ Lucy was so happy that she ate all of her food. She had been waiting for it for a long time because it was so yummy. ⏎ When it was time to go home, Lucy gave her mom a big hug  _(120 tokens)_
- **The dog was very** playful. Every day, the dog would run around and run in circles. One day, the dog's owner took him for a walk.  ⏎ The dog's owner was worried and asked the owner to stop, but the dog just shook his head. The owner said, "No, dog. It's tough to walk." ⏎ So the owner kept driving the dog. The dog was so happy to be with his owner. After a while, the owner was tired but he stopped and said, "We have to go home now." ⏎ The pup wagged his tail and barked in agreement. The owner said goodbye  _(120 tokens)_
- **One day, Lily went to the park and** saw a big tree. She wanted to climb it, but it was difficult. She started to struggle because she didn't know how to manage. She asked her friend, Timmy, to help her. ⏎ Timmy said, "You can do it! Just try, I will help you." He went up the tree and started to climb. He was so happy that he finally managed to climb the tree.  ⏎ Lily was so proud of herself and said to Timmy, "You did a great job! I'm so happy I could manage." Timmy smiled and said, "I'm glad I could lend you  _(120 tokens)_
- **Tom and his mom** were walking down the street when they heard a loud noise. "What is that noise?" asked Tom. "It is a truck and it has a big engine," his mom said. ⏎ Tom was so intrigued. He wanted to know what it was doing. "Mommy, what is that?" he asked. ⏎ "That is a truck, Tom. It tells you stories," his mom said. "It tells you stories about the sounds of the truck making." ⏎ Tom wanted to hear the sound. He asked his mom many questions. "Can you tell me a story about the truck?" ⏎ His mom smiled and said  _(120 tokens)_

### step 2250

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore. One day, she was walking through her garden when she saw something that was yellow. It was a yellow bird, but it was very pretty.  ⏎ Lucy wanted to get closer to see the yellow bird, so she slowly approached the bird. As she got closer, the bird flew away. Lucy was amazed! ⏎ She wanted to own the yellow bird forever so she decided to take it home. She gave it a soft cloth and some some food. She wanted to take care of it and take care of it, so that it  _(120 tokens)_
- **The dog was very** happy. "Woof woof!" the dog barked. "I love to play!" ⏎ Lily saw a big box with a bow on it. She wanted to see what was inside. She put the bow on the dog's collar. The dog saw the bow and opened it. ⏎ Inside the box, there were a lot of colorful toys. Lily and the dog played with the toys all day. They had a lot of fun. Lily saw a toy she liked. It was a small, red ball. The dog and the dog played together until it was time to go home.  _(115 tokens, ended with <|eos|>)_
- **One day, Lily went to the park and** saw a big slide. She wanted to try it, so she ran to the slide and climbed up the steps. She sat on the slide and looked down. She saw a boy who was not being nice. He had a mean face and a loud voice. He said, "This is my slide! Go away!" ⏎ Lily was scared and sad. She wanted to go back to the boy, but he was still on the slide. She wanted to go home, but he was too strong and fast. She sat on the slide and tried to push herself down. But she was too slow and the  _(120 tokens)_
- **Tom and his mom** were busy in the living room. They had many cars, dolls, books and books to see and play with. Tom liked to make noises and pretend he was in his room. ⏎ "Look, mom, I'm getting ready for your new game," Tom said. He put on his shoes and ran to his room. He saw his toy phone on the table. ⏎ "Good job, Tom. You have to record your voice before you go to bed," his mom said. She gave him a hug and a kiss. ⏎ "Thank you, mom. I will record my voice," Tom said. He walked around his  _(120 tokens)_

### step 2500

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore. One day, she was walking through her neighborhood when she suddenly saw a pile of junk in her neighbor's yard. She ran over to get a closer look. ⏎ As she got closer, she noticed the junk had lots of pieces. She was so excited that she couldn't help but reach out and pick them up. But as she did, a gust of wind came out of the junk. ⏎ The little girl had no idea what to do. She started to cry. Suddenly, a friendly man came out of his house and said  _(120 tokens)_
- **The dog was very** playful. Every day, the dog would run around and run. He was always laughing and wagging his tail. ⏎ One day, the dog got very dirty. His fur was all muddy and his fur was very dirty. He decided to throw a bath. He filled up the tub with warm water and began to soak inside the tub. ⏎ But the dog was too lazy to take a bath. He just sat in the tub and began to get really filthy. He was trying to get out, but it was too late. The water had spoiled his fur. He was so embarrassed. ⏎ But then, the dog  _(120 tokens)_
- **One day, Lily went to the park and** saw a big slide. She wanted to try it, but it was difficult to climb. She tried and tried, but she couldn't manage to do it. She started to yell, "I want to go on the slide!" ⏎ A little boy named Timmy saw her and asked, "What's wrong, Lily?" Lily replied, "My mom and I want to try the slide, but it's too hard." Timmy said, "Don't worry, I'll help you." He climbed up the stairs and sat on the slide. He said, "Look, Lily! I can do it!"  _(120 tokens)_
- **Tom and his mom** were in the kitchen. They wanted to make a cake. Mom gave them a big bowl and a knife. Tom wanted to help, but the knife was too sharp. He tried to cut the cake, but he hurt his hand. He cried out loud. ⏎ Mom heard Tom and came to see. She saw the knife and the bowl. She was angry. She said, "Tom, that knife is not for you. It is for grown-ups. You could have hurt yourself. You are not allowed in the kitchen. You are supposed to be in the kitchen." ⏎ Tom felt bad. He did  _(120 tokens)_

### step 2750

- **Once upon a time, there was a little** girl named Lucy. She was three years old and loved to watch the world go by her window. ⏎ One day, Lucy's mom said, "Lucy, let's go outside and watch the birds." Lucy was so excited! She grabbed her mom's hand and they ran outside. ⏎ They saw a big tree and Lucy said, "Mom, I want to go there!" Her mom said, "It's okay, Lucy. We can stay out together and watch the birds." ⏎ So, Lucy and her mom sat on the grass and watched the birds. Lucy was so excited that she almost forgot what her  _(120 tokens)_
- **The dog was very** happy. "The weather is brilliant!" it said. "I like to run and play in the grass." ⏎ One day, the dog got very dirty. His fur was dirty and he had run and played in the mud. He wanted to help his friends, but he had to leave the house. ⏎ The dog's friend, a cat, saw him. "Why did you leave the house?" he asked. The dog explained that he was dirty and needed a bath. The cat said, "I would like to wash you." ⏎ The dog went to the water and washed his fur very well. All  _(120 tokens)_
- **One day, Lily went to the park and** saw a big slide. She wanted to try it, but she was scared. Her mom said, "Don't be scared, Lily. You can do it. Just slide down." Lily took a deep breath and slid down the slide. ⏎ When she got to the bottom, she felt the wind on her face. She looked down and saw a big, orange slide. She took a deep breath and climbed up the stairs. When she reached the top, she saw her mom waiting for her at the bottom. ⏎ "Wow, Lily! You did it! You slide!" her mom said. ⏎ "Thank you,  _(120 tokens)_
- **Tom and his mom** were walking down the street. On the street, they saw a man walking a car. Tom asked his mom, "What's that man doing?" His mom answered, "He's looking for something." ⏎ They kept walking and soon, they saw a man with a big box. Tom asked his mom what he was doing and she said, "He is carrying this box." She smiled and opened it. Inside was a small, black animal. She was a bit scared and a bit nervous. She asked Tom, "Do you know what it is?" ⏎ Tom shook his head and said, "No  _(120 tokens)_

### step 3000

- **Once upon a time, there was a little** girl named Lucy. She was three years old and she loved to explore the world around her. One day, her mommy said to her, "Lucy, let's go for a walk!" ⏎ They walked and walked until they came to a big, wide field. The grass was so green and the sun was so bright. Lucy's mommy said, "Let's go and pick some flowers!" So they started picking the flowers and picking the prettiest ones together. ⏎ After they finished picking, they heard a loud noise behind them. It was a big truck coming towards them! Lucy and her mommy quickly grabbed each  _(120 tokens)_
- **The dog was very** happy. He liked to play with his friends in the park. One day, he saw a big box with a bow on it. He wanted to know what was inside. He thought it was a toy, but it was not a toy. It was a fake toy. ⏎ The dog wanted to see the real thing. He thought it was fun to play with. So, he sat down and started to rub the fake toy with his nose. The fake toy started to shrink! It became smaller and smaller. ⏎ The dog was so amazed. He played with his toy in the park all day. All  _(120 tokens)_
- **One day, Lily went to the park and** saw a big slide. She wanted to try it, but it was difficult to climb. She tried and tried, but she couldn't manage to do it. She felt sad and embarrassed that she couldn't do it. ⏎ Suddenly, Lily saw a little boy who was waiting for his turn to slide down the slide. She asked him if he wanted to come on it with her. The little boy was so excited and said yes. They slid down the slide together and had so much fun. ⏎ From that day on, Lily was no longer embarrassed because she could always ask for help when she needed it.  _(120 tokens)_
- **Tom and his mom** went to the park. They saw a big slide and wanted to go on it. Tom climbed up the ladder and slid down the slide. He felt happy and free. ⏎ But when he reached the bottom, he saw a boy who was very rude. He said, "This is my slide. You have to wait here." Tom did not want to wait. He wanted to go on the slide. He said, "No, this is my slide. You have to go." He pushed the boy and ran away. ⏎ Tom's mom was very angry. She said, "You are not a nice  _(120 tokens)_

## Checkpoints

| step | path | bytes | sha256 |
|---:|---|---:|---|
| 0 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0000000\checkpoint.pt` | 365,033,715 | `72d3f68b216bfb7ceec45eab8fa7f14ec02c65d55cb42c949097309460e0732c` |
| 500 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0000500\checkpoint.pt` | 1,095,092,299 | `ee6a57315073252cc6b0bd8b81f76f206e0aa08997ffb5faaf2a096afe43ed49` |
| 1000 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0001000\checkpoint.pt` | 1,095,092,427 | `f95495a7fb59c2296d40c57d71eae0b71142e38d24950cfc80605b52bbba00d9` |
| 1500 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0001500\checkpoint.pt` | 1,095,092,555 | `31238faa65dfe03cbee4599be5478bfbdace99cf210cb0a0b71b2a7807097e9b` |
| 2000 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0002000\checkpoint.pt` | 1,095,092,747 | `4b82abd92bd57410409635c8278e10cbbe31b7f121cb3cde8a0a6b98172e600c` |
| 2500 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0002500\checkpoint.pt` | 1,095,092,875 | `07083aeff96fcce32183fbcc2354585e074291ebe3c3482a51dc56d59849694c` |
| 3000 | `C:\DaveLM-Training\rnc\runs\rnc1_tinystories_shakedown_001\ckpt_0003000\checkpoint.pt` | 1,095,093,003 | `f695c7d72a18424d4a0d5e2ed139955589e3f4786c64a090538c5ed2055483ae` |
