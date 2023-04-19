from .data import SampleCollection
from .tokenizer import Tokenizer
from .encoder import Encoder
from .decoder import Decoder
from .display import Display
from .dataset import Dataset
from .model import Model
from .loss import Loss
from .trainer import Trainer
from .evaluator import LossEvaluator
from ..util.explain import EXPLAIN
from ..util.split import split_val


def main(config, logger):
    path = config['global']['path']
    logger.info('Loading samples...')
    samples = SampleCollection.load(config['samples']['path'])
    samples = [s for s in samples if s.src.text.strip() and s.dst.text.strip()]
    logger.info(f'Samples: {len(samples)}')
    samples_train, samples_val = split_val(samples, **config['val'])
    
    vocab_size = config['tokenizer']['vocab_size']
    logger.info(f'Training src tokenizer...')
    texts_src = [s.src.text for s in samples]
    tokenizer_src = Tokenizer(**config.get('tokenizer', {}))
    tokenizer_src.train(texts_src)
    tokenizer_src.save(path=path, postfix='src')
    tokenizer_src = Tokenizer.load(path=path, postfix='src', vocab_size=vocab_size)

    logger.info(f'Training dst tokenizer...')
    texts_dst = [s.dst.text for s in samples]
    tokenizer_dst = Tokenizer(**config.get('tokenizer', {}))
    tokenizer_dst.train(texts_dst)
    tokenizer_dst.save(path=path, postfix='dst')
    tokenizer_dst = Tokenizer.load(path=path, postfix='dst', vocab_size=vocab_size)

    encoder = Encoder(src_tokenizer=tokenizer_src, dst_tokenizer=tokenizer_dst, **config.get('encoder', {}))
    decoder = Decoder(src_tokenizer=tokenizer_src, dst_tokenizer=tokenizer_dst)
    display = Display(decoder=decoder)
    dataset_train = Dataset(samples=samples_train, encoder=encoder, display=display)
    dataset_val = Dataset(samples=samples_val, encoder=encoder, display=display)
    EXPLAIN.maybe(dataset_train)

    model = Model(vocab_size=tokenizer_dst.vocab_size)
    loss = Loss()
    evaluator = LossEvaluator(
        dataset=dataset_val,
        model=model,
        loss_fn=loss,
        **config['train']
    )
    trainer = Trainer(
        logger=logger,
        path=path,
        dataset=dataset_train,
        model=model,
        loss_fn=loss,
        evaluator=evaluator,
        **config['train']
    )
    trainer.run()
