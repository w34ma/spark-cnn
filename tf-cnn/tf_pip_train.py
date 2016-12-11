from tf_pip_read import *
from tf_pip_model import *
from config import *
import psutil

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        image, label = train_input(BATCH_SIZE)
        model = cnn_model()
        logits = model.init(image)
        loss = cal_loss(logits, label)
        train_op = train_model(loss, global_step)
        
        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1;
                self.process = psutil.Process(os.getpid())
        
            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)
    
            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                print("time = %f" % duration)
                mem = (self.process.get_memory_info()[0] / float(2 ** 20))
                print("memory usage %f MB" % mem)
                if self._step % 100 == 0:
                    print("acc = %f" % loss_value)
    
                
        with tf.train.MonitoredTrainingSession(hooks=[tf.train.StopAtStepHook(last_step = ITERATION_NUM), tf.train.NanTensorHook(loss), _LoggerHook()], checkpoint_dir = CHECKPOINT_DIR, save_checkpoint_secs = 30) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)     

def main(argv=None):
    if tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.DeleteRecursively(CHECKPOINT_DIR)
    tf.gfile.MakeDirs(CHECKPOINT_DIR)
    train()

if __name__ == '__main__':
    tf.app.run()
